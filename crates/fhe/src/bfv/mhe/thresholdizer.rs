use fhe_math::rq::{Context, Poly, Representation};
use itertools::Itertools;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::FromPrimitive;
use rand::{distributions::Uniform, thread_rng, Rng};
use std::{cmp::min, collections::HashMap, hash::Hash, sync::Arc};

pub type ShamirPublicPoint = u64;

#[derive(Clone)]
pub struct ShamirPolynomial {
	coeffs: Vec<Poly>,
}

#[derive(Clone)]
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
		point: ShamirPublicPoint,
		shamir_poly: &ShamirPolynomial,
	) -> ShamirSecretShare {
		let mut out = Poly::zero(shamir_poly.coeffs[0].ctx(), Representation::PowerBasis);
		let point = BigUint::from_u64(point).unwrap();
		// evaluate the polynomial at secret_point of jth point for generating their
		// respective share
		for i in (1..(shamir_poly.coeffs.len())).rev() {
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

#[derive(Clone)]
pub struct Combiner {
	threshold: usize,
	ctx: Arc<Context>,
	langrange_coeffs: HashMap<ShamirPublicPoint, BigUintDig>,
}

impl Combiner {
	pub fn new(
		ctx: &Arc<Context>,
		threshold: usize,
		all_points: &[ShamirPublicPoint],
		own_point: ShamirPublicPoint,
	) -> Combiner {
		let mut langrange_coeffs = HashMap::new();
		let m = BigUintDig::from_bytes_le(ctx.modulus().clone().to_bytes_le().as_slice());
		all_points.iter().for_each(|p| {
			if p != &own_point {
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
			}
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
		own_share: &ShamirSecretShare,
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
mod tests {
	use std::vec;

	use fhe_math::rq::traits::TryConvertFrom;
	use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};

	use crate::bfv::{
		mhe::{Ckg, Cks},
		BfvParameters, Encoding, Plaintext, PublicKey, SecretKey,
	};

	use super::*;

	#[derive(Clone)]
	struct Party {
		key: SecretKey,
		shamir_public_point: ShamirPublicPoint,
		secret_shares: Vec<ShamirSecretShare>,
		own_share: Option<ShamirSecretShare>,
	}

	#[test]
	fn threshold() {
		let mut rng = thread_rng();
		let n = 10;
		let t = 4;

		let params = Arc::new(BfvParameters::default(6, 8));

		let mut public_points = vec![];
		let mut parties = (0..n)
			.map(|_| {
				let key = SecretKey::random(&params, &mut rng);

				let mut point =
					Thresholdizer::gen_shamir_secret_point(params.ctx_at_level(0).unwrap());
				while public_points.contains(&point) {
					point = Thresholdizer::gen_shamir_secret_point(params.ctx_at_level(0).unwrap());
				}
				public_points.push(point);
				Party {
					key,
					shamir_public_point: point,
					secret_shares: vec![],
					own_share: None,
				}
			})
			.collect_vec();

		for i in (0..parties.len()) {
			let sk = Poly::try_convert_from(
				parties[i].key.coeffs.as_ref(),
				params.ctx_at_level(0).unwrap(),
				false,
				Representation::PowerBasis,
			)
			.unwrap();
			let shamir_poly = Thresholdizer::gen_shamir_polynomial(t, &sk, sk.ctx());

			// gen secret share of ith party for jth party
			for j in 0..parties.len() {
				let share = Thresholdizer::gen_shamir_secret_share(
					parties[j].shamir_public_point,
					&shamir_poly,
				);
				parties[j].secret_shares.push(share);
			}
		}

		parties.iter_mut().for_each(|p| {
			let agg_s = Thresholdizer::agg_shamir_secret_shares(&p.secret_shares);
			p.own_share = Some(agg_s);
		});

		let mut combiners = parties
			.iter()
			.map(|p| {
				Combiner::new(
					params.ctx_at_level(0).unwrap(),
					t,
					&public_points,
					p.shamir_public_point,
				)
			})
			.collect_vec();

		// Collective key generation - Phase 1
		// Consider that only first t parties are active
		let active_points = parties
			.iter()
			.take(t)
			.map(|p| p.shamir_public_point)
			.collect_vec();
		let t_secrets = combiners
			.iter()
			.take(t)
			.enumerate()
			.map(|(index, c)| {
				c.combine(
					&active_points,
					parties[index].shamir_public_point,
					parties[index].own_share.as_ref().unwrap(),
				)
			})
			.collect_vec();

		let crp = Ckg::sample_crp(&params);
		let ckg = Ckg::new(&params, &crp);
		let pk = {
			let ckg_shares = t_secrets
				.iter()
				.map(|ts| {
					let mut s = ts.clone();
					s.change_representation(Representation::Ntt);
					ckg.gen_share_poly(&s)
				})
				.collect_vec();
			let agg_shares = ckg.aggregate_shares(&ckg_shares);
			PublicKey::new_from_ckg(&ckg, &agg_shares)
		};

		// Encrypt message usinf pk generated in collective key generation
		let m = params.plaintext.random_vec(params.degree(), &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
		let ct = pk.try_encrypt(&pt, &mut rng).unwrap();

		// Collective key switching to decrypt ct - Phase 2.
		// Consider that last t parties are active
		parties.reverse();
		combiners.reverse();
		let active_points = parties
			.iter()
			.take(t)
			.map(|p| p.shamir_public_point)
			.collect_vec();
		let t_secrets = combiners
			.iter()
			.take(t)
			.enumerate()
			.map(|(index, c)| {
				c.combine(
					&active_points,
					parties[index].shamir_public_point,
					parties[index].own_share.as_ref().unwrap(),
				)
			})
			.collect_vec();

		let cks = Cks::new(&ct);
		let zero = Poly::zero(ct.c[0].ctx(), Representation::PowerBasis);
		let shares = t_secrets
			.iter()
			.map(|ts| cks.gen_share_poly(ts, &zero))
			.collect_vec();
		let ct_switched = cks.key_switch(&shares);
		let sk_zero = SecretKey {
			coeffs: vec![0i64; params.degree()].into_boxed_slice(),
			par: params.clone(),
		};
		let pt2 = sk_zero.try_decrypt(&ct_switched).unwrap();
		let m2 = Vec::<u64>::try_decode(&pt2, Encoding::simd()).unwrap();

		assert_eq!(m, m2);
	}
}
