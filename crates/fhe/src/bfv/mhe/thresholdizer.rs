use fhe_math::rq::{Context, Poly, Representation};
use itertools::Itertools;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::FromPrimitive;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::{cmp::min, collections::HashMap, hash::Hash, sync::Arc};

pub type ShamirPublicPoint = u64;

pub struct ShamirPolynomial {
	coeffs: Vec<Poly>,
}

pub struct ShamirSecretShare(Poly);

pub struct Thresholdizer {}

impl Thresholdizer {
	/// Draws random value from Z_qmin, where qmin = min(q_1...q_l) for l
	/// modulus in moduli Q. This gives us performance boost in multiplication
	/// of secret points during polynomial evaluation and ensure that resulting
	/// constant polynomial is an exceptional sequence.
	///
	/// Section 3.2 of https://eprint.iacr.org/2022/780
	pub fn gen_shamir_secret_point(ctx: &Arc<Context>) -> ShamirPublicPoint {
		let mut rng = thread_rng();
		let mut q = ctx.moduli().to_vec();
		q.sort();
		// Secret point cannot be 0
		rng.sample(Uniform::new(1, q[0]))
	}

	pub fn gen_shamir_polynomial(
		threshold: usize,
		secret: &Poly,
		ctx: &Arc<Context>,
	) -> ShamirPolynomial {
		debug_assert!(threshold > 1);
		debug_assert!(secret.ctx() == ctx);
		debug_assert!(secret.representation() == &Representation::PowerBasis);

		let mut rng = thread_rng();
		let coeffs = (0..threshold)
			.map(|i| {
				if i == 0 {
					secret.clone()
				} else {
					Poly::random(ctx, Representation::PowerBasis, &mut rng)
				}
			})
			.collect_vec();

		ShamirPolynomial { coeffs }
	}

	pub fn gen_shamir_secret_share(
		secret_point: ShamirPublicPoint,
		shamir_poly: ShamirPolynomial,
	) -> ShamirSecretShare {
		let mut out = Poly::zero(shamir_poly.coeffs[0].ctx(), Representation::PowerBasis);
		let point = BigUint::from_u64(secret_point).unwrap();
		// evaluate the polynomial at secret_point of jth point for generating their
		// respective share
		for i in (shamir_poly.coeffs.len() - 1)..0 {
			out += &shamir_poly.coeffs[i];
			out *= &point;
		}
		out += &shamir_poly.coeffs[0];
		ShamirSecretShare(out)
	}

	pub fn agg_shamir_secret_shares(shares: &[ShamirSecretShare]) -> ShamirSecretShare {
		let mut out = Poly::zero(shares[0].0.ctx(), Representation::PowerBasis);
		shares.iter().for_each(|s| {
			out += &s.0;
		});
		ShamirSecretShare(out)
	}
}

pub struct Combiner {
	threshold: usize,
	ctx: Arc<Context>,
	langrange_coeffs: HashMap<ShamirPublicPoint, BigUintDig>,
}

impl Combiner {
	pub fn new(
		ctx: &Arc<Context>,
		threshold: usize,
		other_points: &[ShamirPublicPoint],
		own_point: ShamirPublicPoint,
	) -> Combiner {
		let mut langrange_coeffs = HashMap::new();
		let m = BigUintDig::from_bytes_le(ctx.modulus().clone().to_bytes_le().as_slice());
		other_points.iter().for_each(|p| {
			let s = {
				if p > &own_point {
					BigUintDig::from_u64(p - own_point).unwrap()
				} else {
					&m - BigUintDig::from_u64(own_point - p).unwrap()
				}
			};
			let s_inv = s.mod_inverse(&m).unwrap().to_biguint().unwrap();
			let v = s_inv * p;
			langrange_coeffs.insert(*p, v);
		});

		Combiner {
			ctx: ctx.clone(),
			threshold,
			langrange_coeffs,
		}
	}

	pub fn combine(
		&self,
		active_points: &[ShamirPublicPoint],
		own_point: ShamirPublicPoint,
		own_share: ShamirSecretShare,
	) -> Poly {
		assert!(active_points.len() >= self.threshold);
		let mut prod = BigUintDig::from_usize(1).unwrap();
		for point in active_points {
			if *point != own_point {
				prod *= self.langrange_coeffs.get(point).unwrap();
			}
		}
		let prod = BigUint::from_bytes_le(prod.to_bytes_le().as_slice());
		&own_share.0 * &prod
	}
}

#[cfg(test)]
mod tests {}
