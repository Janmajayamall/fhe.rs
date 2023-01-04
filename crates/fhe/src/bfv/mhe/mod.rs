use std::{ops::Sub, sync::Arc};

use fhe_math::{
	rns::RnsContext,
	rq::{
		switcher::Switcher, traits::TryConvertFrom, Context, Poly, Representation,
		SubstitutionExponent,
	},
};
use itertools::Itertools;
use rand::thread_rng;

use super::{ciphertext, BfvParameters, Ciphertext, SecretKey};

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
		assert_eq!(ciphertext_ctx, ksk_ctx);
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
		sk: &[i64],
		rlk_eph_sk: &[i64],
	) -> RkgShare {
		let mut rng = thread_rng();

		let mut sk =
			Poly::try_convert_from(sk, &self.ciphertext_ctx, false, Representation::PowerBasis)
				.unwrap();
		let switcher = Switcher::new(&self.ciphertext_ctx, &self.ksk_ctx).unwrap();
		sk = sk.mod_switch_to(&switcher).unwrap();
		let sk_power_basis = sk.clone();
		sk.change_representation(Representation::Ntt);

		let mut rlk_eph_sk =
			Poly::try_convert_from(rlk_eph_sk, &self.ksk_ctx, false, Representation::PowerBasis)
				.unwrap();
		rlk_eph_sk.change_representation(Representation::Ntt);

		let rns = RnsContext::new(self.ciphertext_ctx.moduli()).unwrap();
		let rkg_shares: Vec<[Poly; 2]> = (0..self.crps.len())
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
				e0 -= &(&a * &rlk_eph_sk);

				let garner = rns.get_garner(i).unwrap();
				let mut s_g = &sk_power_basis * garner;
				s_g.change_representation(Representation::Ntt);
				e0 += &s_g;

				let e1 = Poly::small(
					&self.ksk_ctx,
					Representation::Ntt,
					params.variance,
					&mut rng,
				)
				.unwrap();
				a *= &sk;
				a += &e1;

				[e0, a]
			})
			.collect();
		RkgShare { share: rkg_shares }
	}

	pub fn gen_share_round2(
		&self,
		params: &Arc<BfvParameters>,
		agg_shares_round1: &[[Poly; 2]],
		sk: &[i64],
		rlk_eph_sk: &[i64],
	) -> RkgShare {
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
		rlk_eph_sub_sk -= &sk;

		let s = (0..self.crps.len())
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
			.collect_vec();

		RkgShare { share: s }
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

pub struct Rtg {
	pub(crate) par: Arc<BfvParameters>,
	pub(crate) ciphertext_level: usize,
	pub(crate) ctx_ciphertext: Arc<Context>,

	pub(crate) ksk_level: usize,
	pub(crate) ctx_ksk: Arc<Context>,

	pub(crate) crps: Vec<Poly>,
	pub(crate) element: SubstitutionExponent,
}

pub struct RtgShare {
	share: Vec<Poly>,
}

impl Rtg {
	pub fn new(
		par: &Arc<BfvParameters>,
		ciphertext_level: usize,
		ksk_level: usize,
		crps: &[Poly],
		element: SubstitutionExponent,
	) -> Rtg {
		let ctx_ciphertext = par.ctx_at_level(ciphertext_level).unwrap().clone();
		let ctx_ksk = par.ctx_at_level(ksk_level).unwrap().clone();

		Rtg {
			par: par.clone(),
			ciphertext_level,
			ctx_ciphertext,
			ksk_level,
			ctx_ksk,
			crps: crps.to_vec(),
			element,
		}
	}

	pub fn sample_crps(
		par: &Arc<BfvParameters>,
		ciphertext_level: usize,
		ksk_level: usize,
	) -> Vec<Poly> {
		let ctx_ciphertext = par.ctx_at_level(ciphertext_level).unwrap();
		let ctx_ksk = par.ctx_at_level(ksk_level).unwrap();
		let mut rng = thread_rng();
		(0..ctx_ciphertext.moduli().len())
			.map(|_| Poly::random(ctx_ksk, Representation::Ntt, &mut rng))
			.collect_vec()
	}

	pub fn gen_share(&self, sk: &SecretKey) -> RtgShare {
		let sk_poly = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			&self.ctx_ciphertext,
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		let sk_sub = sk_poly.substitute(&self.element).unwrap();
		let switcher = Switcher::new(&self.ctx_ciphertext, &self.ctx_ksk).unwrap();
		let sk_sub_switched = sk_sub.mod_switch_to(&switcher).unwrap();

		let mut sk_poly = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			&self.ctx_ksk,
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		sk_poly.change_representation(Representation::Ntt);

		let mut rng = thread_rng();

		let rns = RnsContext::new(self.ctx_ciphertext.moduli()).unwrap();
		let share = (0..self.crps.len())
			.map(|i| {
				let mut e = Poly::small(
					&self.ctx_ksk,
					Representation::Ntt,
					self.par.variance,
					&mut rng,
				)
				.unwrap();
				e -= &(&self.crps[i] * &sk_poly);

				let garner = rns.get_garner(i).unwrap();
				let mut garner = &sk_sub_switched * garner;
				garner.change_representation(Representation::Ntt);
				e += &garner;

				e
			})
			.collect();

		RtgShare { share }
	}

	pub fn aggregate_shares(&self, shares: &[RtgShare]) -> Vec<Poly> {
		let mut agg = vec![Poly::zero(&self.ctx_ksk, Representation::Ntt); self.crps.len()];
		shares.iter().for_each(|s| {
			(0..self.crps.len()).for_each(|i| {
				agg[i] += &s.share[i];
			})
		});
		agg
	}
}

struct CksShare {
	share: Poly,
}

struct Cks {
	ct: Ciphertext,
}

impl Cks {
	pub fn gen_share(&self, sk: &SecretKey, to_key: &[i64]) -> CkgShare {
		let sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			self.ct.c[0].ctx(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		let to_key = Poly::try_convert_from(
			to_key,
			self.ct.c[0].ctx(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		let mut s = &sk - &to_key;
		s.change_representation(Representation::Ntt);

		CkgShare {
			share: &s * &self.ct.c[1],
		}
	}

	pub fn key_switch(&self, shares: &[CksShare]) -> Ciphertext {
		let mut agg = Poly::zero(self.ct.c[0].ctx(), Representation::Ntt);
		shares.iter().for_each(|s| {
			agg += &s.share;
		});
		agg += &self.ct.c[0];

		Ciphertext {
			par: self.ct.par.clone(),
			seed: self.ct.seed,
			c: vec![agg, self.ct.c[0].clone()],
			level: self.ct.level,
		}
	}
}

#[cfg(test)]
mod tests {
	use fhe_math::zq::Modulus;
	use fhe_traits::{FheDecoder, FheEncoder, FheEncrypter};

	use super::*;
	use crate::bfv::{
		keys::GaloisKey, BfvParameters, Ciphertext, Encoding, Plaintext, PublicKey,
		RelinearizationKey,
	};

	fn try_decrypt(par: &Arc<BfvParameters>, parties: &[Party], ct: &Ciphertext) -> Plaintext {
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
		let mut ski = sk.clone();
		let mut m_scaled = ct.c[0].clone();
		for i in 1..ct.c.len() {
			let t = &ski * &ct.c[i];
			m_scaled += &t;
			ski *= &sk;
		}
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

		let mut poly_ntt =
			Poly::try_convert_from(&w, ct.c[0].ctx(), false, Representation::PowerBasis).unwrap();
		poly_ntt.change_representation(Representation::Ntt);

		Plaintext {
			par: par.clone(),
			value: w.into(),
			encoding: None,
			poly_ntt,
			level: ct.level,
		}
	}

	fn gen_ckg(params: &Arc<BfvParameters>, parties: &[Party]) -> PublicKey {
		let ckg_crp = Ckg::sample_crp(&params);
		let ckg = Ckg::new(&params, &ckg_crp);

		let ckg_shares = parties
			.iter()
			.map(|p| ckg.gen_share(&ckg_crp, &p.key))
			.collect_vec();

		let ckg_agg_shares = ckg.aggregate_shares(&ckg_shares);
		PublicKey::new_from_ckg(&ckg, &ckg_agg_shares, &ckg_crp)
	}

	#[test]
	fn ckg() {
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
		let ckg_pk = gen_ckg(&params, &parties);

		let m = params.plaintext.random_vec(8, &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::poly(), &params).unwrap();

		let ct = ckg_pk.try_encrypt(&pt, &mut rng).unwrap();
		let pt_decrypted = try_decrypt(&params, &parties, &ct);

		assert_eq!(
			Vec::<u64>::try_decode(&pt_decrypted, Encoding::poly()).unwrap(),
			m
		);
	}

	#[test]
	fn rkg() {
		let mut rng = thread_rng();
		let params = Arc::new(BfvParameters::default(10, 8));
		let no_of_parties = 4;

		let parties = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();
		let crps = Rkg::sample_crps(&params, 0, 0);
		let rkg = Rkg::new(&params, 0, 0, &crps);

		// round1
		let r1_shares = parties
			.iter()
			.map(|p| {
				rkg.gen_share_round1(
					&params,
					p.key.coeffs.as_ref(),
					p.rlk_eph_key.coeffs.as_ref(),
				)
			})
			.collect_vec();
		let r1_agg_shares = rkg.aggregate_shares(&r1_shares);
		let r2_shares = parties
			.iter()
			.map(|p| {
				rkg.gen_share_round2(
					&params,
					&r1_agg_shares,
					p.key.coeffs.as_ref(),
					p.rlk_eph_key.coeffs.as_ref(),
				)
			})
			.collect_vec();
		let r2_agg_shares = rkg.aggregate_shares(&r2_shares);
		let rlk = RelinearizationKey::new_from_rkg(&rkg, &r2_agg_shares, &r1_agg_shares);

		// collective key generation
		let pk = gen_ckg(&params, &parties);

		let m1 = params.plaintext.random_vec(8, &mut rng);
		let m2 = params.plaintext.random_vec(8, &mut rng);
		let mut m1_m2 = m1.clone();
		params.plaintext.mul_vec(&mut m1_m2, &m2);

		let pt1 = Plaintext::try_encode(&m1, Encoding::simd(), &params).unwrap();
		let pt2 = Plaintext::try_encode(&m2, Encoding::simd(), &params).unwrap();
		let ct1 = pk.try_encrypt(&pt1, &mut rng).unwrap();
		let ct2 = pk.try_encrypt(&pt2, &mut rng).unwrap();

		let mut ct3 = &ct1 * &ct2;
		rlk.relinearizes(&mut ct3).unwrap();

		let pt_decrypted = try_decrypt(&params, &parties, &ct3);

		assert_eq!(
			Vec::<u64>::try_decode(&pt_decrypted, Encoding::simd()).unwrap(),
			m1_m2
		);
	}

	#[test]
	fn rtg() {
		let mut rng = thread_rng();
		let params = Arc::new(BfvParameters::default(10, 8));
		let no_of_parties = 4;

		let parties = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();

		let crps = Rtg::sample_crps(&params, 0, 0);
		let element = SubstitutionExponent::new(params.ctx_at_level(0).unwrap(), 3).unwrap();
		let rtg = Rtg::new(&params, 0, 0, &crps, element);
		let shares = parties.iter().map(|p| rtg.gen_share(&p.key)).collect_vec();
		let agg_shares = rtg.aggregate_shares(&shares);
		let galois_key = GaloisKey::new_from_rtg(&rtg, &agg_shares);

		let pk = gen_ckg(&params, &parties);
		let m = params.plaintext.random_vec(8, &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
		let ct = pk.try_encrypt(&pt, &mut rng).unwrap();

		let ct_sub = galois_key.relinearize(&ct).unwrap();
		let m_sub =
			Vec::<u64>::try_decode(&try_decrypt(&params, &parties, &ct_sub), Encoding::simd())
				.unwrap();

		assert_eq!(&m[1..4], &m_sub[..3]);
		assert_eq!(&m[0], &m_sub[3]);
		assert_eq!(&m[5..], &m_sub[4..7]);
		assert_eq!(&m[4], &m_sub[7]);
	}
}
