use std::sync::Arc;

use fhe_math::{
	rns::RnsContext,
	rq::{switcher::Switcher, traits::TryConvertFrom, Context, Poly, Representation},
};
use itertools::Itertools;
use rand::thread_rng;

use super::{BfvParameters, SecretKey};

struct Party {
	key: SecretKey,
	rlk_eph_key: SecretKey,
}

struct CkgShare {
	share: Poly,
}

pub struct Ckg {
	pub(crate) par: Arc<BfvParameters>,
	pub(crate) crp: Poly,
}
impl Ckg {
	pub fn new(par: &Arc<BfvParameters>, crp: &Poly) -> Ckg {
		Ckg {
			par: par.clone(),
			crp: crp.clone(),
		}
	}

	/// pass the seeded rng here to get the common reference polynomial p1
	pub fn sample_crp(par: &Arc<BfvParameters>) -> Poly {
		let mut rng = thread_rng();
		Poly::random(par.ctx_at_level(0).unwrap(), Representation::Ntt, &mut rng)
	}

	/// Generates ith party's share to ideal pk.
	/// Caculates crp * sk_i + e_i for party i
	///
	/// Note that `cpk` (Common Public Key) is (p0, p1).
	/// `p1` is randomly sampled crp from gaussian distribution.
	/// p0 is aggregation of shares from each party
	///
	/// p0 = Summation(p_0i) for i in 0..N
	/// where p_0i = -crp * sk_i + e_i
	fn gen_share(&self, crp: &Poly, sk: &SecretKey) -> CkgShare {
		let mut rng = thread_rng();
		let e = Poly::small(
			self.par.ctx_at_level(0).unwrap(),
			Representation::Ntt,
			self.par.variance,
			&mut rng,
		)
		.unwrap();

		let mut sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			self.par.ctx_at_level(0).unwrap(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		sk.change_representation(Representation::Ntt);
		sk *= &(-crp);
		sk += &e;

		CkgShare { share: sk }
	}

	/// Aggregates shares of all parties and returns
	/// ideal public key
	///
	/// p0 = Summation(p_0i) for i in 0..N
	fn aggregate_shares(&self, shares: &[CkgShare]) -> Poly {
		let mut agg = Poly::zero(self.par.ctx_at_level(0).unwrap(), Representation::Ntt);
		for sh in shares.iter() {
			debug_assert!(sh.share.representation() == &Representation::Ntt);
			agg += &sh.share;
		}
		agg
	}
}

pub struct RkgShare {
	share: Vec<[Poly; 2]>,
}

/// Relinearization key generation
///
/// Collective Rkg is of the form:
/// (s^2*g - s*b + s*e0 + e1 + u*e2 + e3, b)
/// where b = s*a + e2
///
/// Rlk is generated in two step process
/// 1. Each party generates h0_i, h1_i.
/// h0_i = -u_i*a + s_i*g + e0_i
/// h1_i = s_i*a + e1_i
///
/// Think of (1) as each party generating a pseudo encryption of
/// their secret share s_i under ephemeral key u_i
///
/// Parties send each other their respective shares,after which each party
/// calculates
/// h0 = Summation(h0_i) for i in 0..N
/// h1 = Summation(h1_i) for i in 0..N
///
/// 2. Each party generates (h0'_i, h1'_i)
/// h0'_i = s_i*h0 + e2_i
/// h1'_i = (u_i - s_i)*h1 + e3_i
/// Parties share h0'_i, h1'_i and calculate following to
/// obtain rlk
/// h0' = Summation(h0'_i)
/// h1' = Summation(h1'_i)
/// rlk = (h0' + h1', h1)
pub struct Rkg {
	pub(crate) par: Arc<BfvParameters>,

	pub(crate) ciphertext_level: usize,
	pub(crate) ciphertext_ctx: Arc<Context>,

	pub(crate) ksk_level: usize,
	pub(crate) ksk_ctx: Arc<Context>,

	pub(crate) crps: Vec<Poly>,
}

impl Rkg {
	pub fn new(
		params: &Arc<BfvParameters>,
		ciphertext_level: usize,
		key_level: usize,
		crps: &[Poly],
	) -> Rkg {
		let ciphertext_ctx = params.ctx[ciphertext_level].clone();
		let ksk_ctx = params.ctx[key_level].clone();

		Rkg {
			par: params.clone(),
			ciphertext_level,
			ciphertext_ctx,
			ksk_level: key_level,
			ksk_ctx,
			crps: crps.to_vec(),
		}
	}

	pub fn sample_crps(
		params: &Arc<BfvParameters>,
		ciphertext_level: usize,
		key_level: usize,
	) -> Vec<Poly> {
		let mut rng = thread_rng();
		let ciphertext_ctx = params.ctx[ciphertext_level].clone();
		let ksk_ctx = params.ctx[key_level].clone();

		(0..ciphertext_ctx.moduli().len())
			.into_iter()
			.map(|_| Poly::random(&ksk_ctx, Representation::Ntt, &mut rng))
			.collect()
	}

	pub fn gen_share_round1(
		&self,
		params: &Arc<BfvParameters>,
		sk: &[u64],
		rlk_eph_sk: &[u64],
	) -> RkgShare {
		let mut rng = thread_rng();

		let mut sk =
			Poly::try_convert_from(sk, &self.ciphertext_ctx, false, Representation::PowerBasis)
				.unwrap();
		let switcher = Switcher::new(&self.ciphertext_ctx, &self.ksk_ctx).unwrap();
		sk = sk.mod_switch_to(&switcher).unwrap();
		sk.change_representation(Representation::Ntt);

		let mut rlk_eph_sk =
			Poly::try_convert_from(rlk_eph_sk, &self.ksk_ctx, false, Representation::PowerBasis)
				.unwrap();
		rlk_eph_sk.change_representation(Representation::Ntt);
		rlk_eph_sk = -rlk_eph_sk;

		let rns = RnsContext::new(self.ciphertext_ctx.moduli()).unwrap();
		let rkg_shares = (0..self.crps.len())
			.into_iter()
			.map(|i| {
				let mut e0 = Poly::small(
					&self.ksk_ctx,
					Representation::Ntt,
					params.variance,
					&mut rng,
				)
				.unwrap();
				let mut a = self.crps[i].clone();
				e0 += &(&a * &rlk_eph_sk);

				let mut garner = Poly::try_convert_from(
					&[rns.get_garner(i).unwrap().clone()],
					&self.ksk_ctx,
					false,
					Representation::Ntt,
				)
				.unwrap();
				garner *= &sk;
				garner += &e0;

				let e1 = Poly::small(
					&self.ksk_ctx,
					Representation::Ntt,
					params.variance,
					&mut rng,
				)
				.unwrap();
				a *= &sk;
				a += &e1;

				[garner, a]
			})
			.collect();

		RkgShare { share: rkg_shares }
	}

	pub fn gen_share_round2(
		&self,
		params: &Arc<BfvParameters>,
		agg_shares_round1: Vec<[Poly; 2]>,
		sk: &[u64],
		rlk_eph_sk: &[u64],
	) -> Vec<[Poly; 2]> {
		let mut rng = thread_rng();

		let mut sk =
			Poly::try_convert_from(sk, &self.ciphertext_ctx, false, Representation::PowerBasis)
				.unwrap();
		let switcher = Switcher::new(&self.ciphertext_ctx, &self.ksk_ctx).unwrap();
		sk = sk.mod_switch_to(&switcher).unwrap();
		sk.change_representation(Representation::Ntt);

		let mut rlk_eph_sub_sk =
			Poly::try_convert_from(rlk_eph_sk, &self.ksk_ctx, false, Representation::PowerBasis)
				.unwrap();
		rlk_eph_sub_sk.change_representation(Representation::Ntt);
		rlk_eph_sub_sk = -rlk_eph_sub_sk;

		(0..self.crps.len())
			.into_iter()
			.map(|i| {
				let mut e0 = Poly::small(
					&self.ksk_ctx,
					Representation::Ntt,
					params.variance,
					&mut rng,
				)
				.unwrap();
				e0 += &(&sk * &agg_shares_round1[i][0]);

				let mut e1 = Poly::small(
					&self.ksk_ctx,
					Representation::Ntt,
					params.variance,
					&mut rng,
				)
				.unwrap();
				e1 += &(&rlk_eph_sub_sk * &agg_shares_round1[i][1]);

				[e0, e1]
			})
			.collect()
	}

	pub fn aggregate_shares(&self, rkg_shares: &[RkgShare]) -> Vec<[Poly; 2]> {
		(0..self.crps.len())
			.into_iter()
			.map(|i| {
				let mut h0 = Poly::zero(&self.ksk_ctx, Representation::Ntt);
				let mut h1 = Poly::zero(&self.ksk_ctx, Representation::Ntt);
				rkg_shares.iter().for_each(|share| {
					h0 += &share.share[i][0];
					h1 += &share.share[i][1];
				});
				[h0, h1]
			})
			.collect_vec()
	}
}

#[cfg(test)]
mod tests {
	use fhe_math::zq::Modulus;
	use fhe_traits::{FheEncoder, FheEncrypter};

	use super::*;
	use crate::bfv::{BfvParameters, Ciphertext, Encoding, Plaintext, PublicKey};

	fn try_decrypt(par: &Arc<BfvParameters>, parties: &[Party], ct: &Ciphertext) -> Vec<u64> {
		// construct ideal sk
		let mut sk = Poly::zero(par.ctx_at_level(0).unwrap(), Representation::PowerBasis);
		parties.iter().for_each(|p| {
			let ski = Poly::try_convert_from(
				p.key.coeffs.as_ref(),
				par.ctx_at_level(0).unwrap(),
				false,
				Representation::PowerBasis,
			)
			.unwrap();
			sk += &ski;
		});
		sk.change_representation(Representation::Ntt);
		let mut m_scaled = &ct.c[0] + &(&sk * &ct.c[1]);
		m_scaled.change_representation(Representation::PowerBasis);

		let d = m_scaled.scale(&par.scalers[0]).unwrap();
		let v = Vec::<u64>::from(d.as_ref())
			.iter()
			.map(|vi| *vi + par.plaintext.modulus())
			.collect_vec();
		let mut w = v[..par.degree()].to_vec();
		let q = Modulus::new(par.moduli[0]).unwrap();
		q.reduce_vec(&mut w);
		par.plaintext.reduce_vec(&mut w);
		w
	}

	#[test]
	fn public_test() {
		let mut rng = thread_rng();
		let params = Arc::new(BfvParameters::default(10, 8));
		let no_of_parties = 4;

		let parties = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();

		// Collective key generation
		let ckg_crp = Ckg::sample_crp(&params);
		let ckg = Ckg::new(&params, &ckg_crp);

		let ckg_shares = parties
			.iter()
			.map(|p| ckg.gen_share(&ckg_crp, &p.key))
			.collect_vec();

		let ckg_agg_shares = ckg.aggregate_shares(&ckg_shares);
		let ckg_pk = PublicKey::new_from_ckg(&ckg, &ckg_agg_shares, &ckg_crp);

		let m = params.plaintext.random_vec(8, &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::poly(), &params).unwrap();

		let ct = ckg_pk.try_encrypt(&pt, &mut rng).unwrap();
		let m_decrypted = try_decrypt(&params, &parties, &ct);

		assert_eq!(m_decrypted, m);
	}
}
