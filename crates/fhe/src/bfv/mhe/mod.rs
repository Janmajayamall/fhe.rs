use fhe_math::{
	rns::RnsContext,
	rq::{
		switcher::Switcher, traits::TryConvertFrom, Context, Poly, Representation,
		SubstitutionExponent,
	},
};
use fhe_util::sample_vec_cbd;
use itertools::Itertools;
use rand::thread_rng;
use std::sync::Arc;
pub mod thresholdizer;

use super::{BfvParameters, Ciphertext, PublicKey, SecretKey};

pub struct Party {
	key: SecretKey,
	rlk_eph_key: SecretKey,
}

pub struct CkgShare {
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
	fn gen_share_poly(&self, sk: &Poly) -> CkgShare {
		let mut rng = thread_rng();
		let e = Poly::small(
			self.par.ctx_at_level(0).unwrap(),
			Representation::Ntt,
			self.par.variance,
			&mut rng,
		)
		.unwrap();

		// let mut sk = Poly::try_convert_from(
		// 	sk.coeffs.as_ref(),
		// 	self.par.ctx_at_level(0).unwrap(),
		// 	false,
		// 	Representation::PowerBasis,
		// )
		// .unwrap();
		// sk.change_representation(Representation::Ntt);
		let mut a = -(&self.crp);
		a *= sk;
		a += &e;

		CkgShare { share: a }
	}

	pub fn gen_share(&self, sk: &SecretKey) -> CkgShare {
		let mut sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			self.par.ctx_at_level(0).unwrap(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		sk.change_representation(Representation::Ntt);
		self.gen_share_poly(&sk)
	}

	/// Aggregates shares of all parties and returns
	/// ideal public key
	///
	/// p0 = Summation(p_0i) for i in 0..N
	pub fn aggregate_shares(&self, shares: &[CkgShare]) -> Poly {
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
		// FIXME: Figure out why switching outputs incorrect values for RLK.
		assert!(ciphertext_level == key_level);
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
		sk: &SecretKey,
		rlk_eph_sk: &SecretKey,
	) -> RkgShare {
		let mut rng = thread_rng();

		let mut sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			&self.ciphertext_ctx,
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		let switcher = Switcher::new(&self.ciphertext_ctx, &self.ksk_ctx).unwrap();
		sk = sk.mod_switch_to(&switcher).unwrap();
		let sk_power_basis = sk.clone();
		sk.change_representation(Representation::Ntt);

		let mut rlk_eph_sk = Poly::try_convert_from(
			rlk_eph_sk.coeffs.as_ref(),
			&self.ksk_ctx,
			false,
			Representation::PowerBasis,
		)
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
		sk: &SecretKey,
		rlk_eph_sk: &SecretKey,
	) -> RkgShare {
		let mut rng = thread_rng();

		let mut sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			&self.ciphertext_ctx,
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		let switcher = Switcher::new(&self.ciphertext_ctx, &self.ksk_ctx).unwrap();
		sk = sk.mod_switch_to(&switcher).unwrap();
		sk.change_representation(Representation::Ntt);

		let mut rlk_eph_sub_sk = Poly::try_convert_from(
			rlk_eph_sk.coeffs.as_ref(),
			&self.ksk_ctx,
			false,
			Representation::PowerBasis,
		)
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

pub struct CksShare {
	share: Poly,
}

pub struct Cks {
	ct: Ciphertext,
}

impl Cks {
	pub fn new(ct: &Ciphertext) -> Cks {
		Cks { ct: ct.clone() }
	}

	pub fn gen_share_poly(&self, sk: &Poly, to_key: &Poly) -> CksShare {
		debug_assert!(to_key.representation() == sk.representation());
		let mut rng = thread_rng();

		let mut s = sk - to_key;
		s.change_representation(Representation::Ntt);
		s *= &self.ct.c[1];

		// FIXME: How does smudging sigma relate to ct sigma?
		// The paper defines relation as s_sigma^2 = 2^lambda * sigma, for security
		// perimeter lambda. But implementation in lattigo sets s_sigma to 3.19, with
		// default sigma 3.2
		let e = Poly::small(self.ct.c[0].ctx(), Representation::Ntt, 12, &mut rng).unwrap();
		s += &e;

		CksShare { share: s }
	}

	pub fn gen_share(&self, sk: &SecretKey, to_key: &Poly) -> CksShare {
		debug_assert!(to_key.representation() == &Representation::PowerBasis);
		let sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			self.ct.c[0].ctx(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();

		self.gen_share_poly(&sk, to_key)
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

pub struct PcksShare {
	share: [Poly; 2],
}

pub struct Pcks {
	to_pk: PublicKey,
	ct: Ciphertext,
}

impl Pcks {
	pub fn new(ct: &Ciphertext, to_pk: &PublicKey) -> Pcks {
		Pcks {
			to_pk: to_pk.clone(),
			ct: ct.clone(),
		}
	}

	pub fn gen_share(&self, sk: &SecretKey) -> PcksShare {
		let mut sk = Poly::try_convert_from(
			sk.coeffs.as_ref(),
			self.ct.c[0].ctx(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		sk.change_representation(Representation::Ntt);
		sk *= &self.ct.c[1];

		let mut rng = thread_rng();
		let u_coeffs =
			sample_vec_cbd(self.ct.par.degree(), self.ct.par.variance, &mut rng).unwrap();
		let mut u = Poly::try_convert_from(
			&u_coeffs,
			self.ct.c[0].ctx(),
			false,
			Representation::PowerBasis,
		)
		.unwrap();
		u.change_representation(Representation::Ntt);
		let e0 = Poly::small(self.ct.c[0].ctx(), Representation::Ntt, 12, &mut rng).unwrap();
		sk += &e0;
		sk += &(&u * &self.to_pk.c.c[0]);

		let e1 = Poly::small(
			self.ct.c[0].ctx(),
			Representation::Ntt,
			self.ct.par.variance,
			&mut rng,
		)
		.unwrap();
		u *= &self.to_pk.c.c[1];
		u += &e1;

		PcksShare { share: [sk, u] }
	}

	pub fn key_switch(&self, shares: &[PcksShare]) -> Ciphertext {
		let mut h0 = Poly::zero(self.ct.c[0].ctx(), Representation::Ntt);
		let mut h1 = Poly::zero(self.ct.c[0].ctx(), Representation::Ntt);
		shares.iter().for_each(|s| {
			h0 += &s.share[0];
			h1 += &s.share[1];
		});
		h0 += &self.ct.c[0];

		Ciphertext {
			par: self.ct.par.clone(),
			seed: None,
			c: vec![h0, h1],
			level: self.ct.level,
		}
	}
}

#[cfg(test)]
mod tests {
	use std::vec;

	use fhe_math::zq::Modulus;
	use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
	use itertools::izip;

	use super::*;
	use crate::bfv::{
		keys::GaloisKey, BfvParameters, Ciphertext, Encoding, Plaintext, PublicKey,
		RelinearizationKey,
	};

	fn try_decrypt(par: &Arc<BfvParameters>, parties: &[Party], ct: &Ciphertext) -> Plaintext {
		// construct ideal sk
		let mut sk = Poly::zero(
			par.ctx_at_level(ct.level).unwrap(),
			Representation::PowerBasis,
		);
		parties.iter().for_each(|p| {
			let ski = Poly::try_convert_from(
				p.key.coeffs.as_ref(),
				par.ctx_at_level(ct.level).unwrap(),
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

		let mut d = m_scaled.scale(&par.scalers[ct.level]).unwrap();
		d.change_representation(Representation::PowerBasis);

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
		let ckg_crp = Ckg::sample_crp(params);
		let ckg = Ckg::new(params, &ckg_crp);

		let ckg_shares = parties.iter().map(|p| ckg.gen_share(&p.key)).collect_vec();

		let ckg_agg_shares = ckg.aggregate_shares(&ckg_shares);
		PublicKey::new_from_ckg(&ckg, &ckg_agg_shares)
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
		let crps = Rkg::sample_crps(&params, 2, 2);
		let rkg = Rkg::new(&params, 2, 2, &crps);

		// round1
		let r1_shares = parties
			.iter()
			.map(|p| rkg.gen_share_round1(&params, &p.key, &p.rlk_eph_key))
			.collect_vec();
		let r1_agg_shares = rkg.aggregate_shares(&r1_shares);
		let r2_shares = parties
			.iter()
			.map(|p| rkg.gen_share_round2(&params, &r1_agg_shares, &p.key, &p.rlk_eph_key))
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
		let mut ct1 = pk.try_encrypt(&pt1, &mut rng).unwrap();
		let mut ct2 = pk.try_encrypt(&pt2, &mut rng).unwrap();
		// ct1.mod_switch_to_next_level();
		// ct2.mod_switch_to_next_level();

		let mut ct3 = &ct1 * &ct2;
		ct3.mod_switch_to_next_level();
		ct3.mod_switch_to_next_level();
		rlk.relinearizes(&mut ct3).unwrap();

		let pt_decrypted = try_decrypt(&params, &parties, &ct3);

		assert_eq!(
			Vec::<u64>::try_decode(&pt_decrypted, Encoding::simd_at_level(0)).unwrap(),
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

		let crps = Rtg::sample_crps(&params, 1, 0);
		let element = SubstitutionExponent::new(params.ctx_at_level(0).unwrap(), 3).unwrap();
		let rtg = Rtg::new(&params, 1, 0, &crps, element);
		let shares = parties.iter().map(|p| rtg.gen_share(&p.key)).collect_vec();
		let agg_shares = rtg.aggregate_shares(&shares);
		let galois_key = GaloisKey::new_from_rtg(&rtg, &agg_shares);

		let pk = gen_ckg(&params, &parties);
		let m = params.plaintext.random_vec(8, &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
		let mut ct = pk.try_encrypt(&pt, &mut rng).unwrap();
		ct.mod_switch_to_next_level();

		let ct_sub = galois_key.relinearize(&ct).unwrap();
		let m_sub =
			Vec::<u64>::try_decode(&try_decrypt(&params, &parties, &ct_sub), Encoding::simd())
				.unwrap();

		assert_eq!(&m[1..4], &m_sub[..3]);
		assert_eq!(&m[0], &m_sub[3]);
		assert_eq!(&m[5..], &m_sub[4..7]);
		assert_eq!(&m[4], &m_sub[7]);
	}

	#[test]
	fn cks() {
		let mut rng = thread_rng();
		let params = Arc::new(BfvParameters::default(10, 8));
		let no_of_parties = 4;

		let parties = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();

		let pk = gen_ckg(&params, &parties);

		let m = params.plaintext.random_vec(8, &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
		let ct = pk.try_encrypt(&pt, &mut rng).unwrap();

		let cks = Cks::new(&ct);

		// decrypt
		let zero_key = Poly::zero(ct.c[0].ctx(), Representation::PowerBasis);
		let shares = parties
			.iter()
			.map(|p| cks.gen_share(&p.key, &zero_key))
			.collect_vec();
		let ct_switched = cks.key_switch(&shares);
		let zero_sk = SecretKey {
			par: params.clone(),
			coeffs: vec![0i64; params.degree()].into_boxed_slice(),
		};
		let m1 = Vec::<u64>::try_decode(
			&zero_sk.try_decrypt(&ct_switched).unwrap(),
			Encoding::simd(),
		)
		.unwrap();
		assert_eq!(m, m1);

		// Some other random collectively known key
		let parties2 = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();

		let shares = izip!(parties.iter(), parties2.iter())
			.map(|(p1, p2)| {
				cks.gen_share(
					&p1.key,
					&Poly::try_convert_from(
						p2.key.coeffs.as_ref(),
						ct.c[0].ctx(),
						false,
						Representation::PowerBasis,
					)
					.unwrap(),
				)
			})
			.collect_vec();
		let ct2 = cks.key_switch(&shares);
		let sk2_ideal = {
			let mut s = Poly::zero(params.ctx_at_level(0).unwrap(), Representation::PowerBasis);
			parties2.iter().for_each(|p| {
				let poly = Poly::try_convert_from(
					p.key.coeffs.as_ref(),
					s.ctx(),
					false,
					Representation::PowerBasis,
				)
				.unwrap();
				s += &poly;
			});
			s.change_representation(Representation::Ntt);
			s
			// let rns = RnsContext::new(s.ctx().moduli()).unwrap();
			// let b = s
			// 	.coefficients()
			// 	.axis_iter(Axis(1))
			// 	.map(|rests| rns.lift(rests))
			// 	.collect_vec();
			// let b = b
			// 	.iter()
			// 	.map(|v| {
			// 		if *v >= rns.modulus().div(2u8) {
			// 			-((rns.modulus() - v).to_i64().unwrap())
			// 		} else {
			// 			v.to_i64().unwrap()
			// 		}
			// 	})
			// 	.collect_vec()
			// 	.into_boxed_slice();
			// b
		};

		let mut m2 = &ct2.c[0] + &(&sk2_ideal * &ct2.c[1]);
		m2.change_representation(Representation::PowerBasis);
		let m2 = m2.scale(&params.scalers[ct2.level]).unwrap();
		let mut m2 = Vec::<u64>::from(&m2)
			.iter()
			.map(|v| v + params.plaintext.modulus())
			.collect_vec();
		Modulus::new(params.moduli[0]).unwrap().reduce_vec(&mut m2);
		params.plaintext.reduce_vec(&mut m2);

		let mut poly_ntt =
			Poly::try_convert_from(&m2, ct.c[0].ctx(), false, Representation::PowerBasis).unwrap();
		poly_ntt.change_representation(Representation::Ntt);

		let pt = Plaintext {
			par: params.clone(),
			value: m2.into(),
			encoding: None,
			poly_ntt,
			level: ct.level,
		};
		let m2 = Vec::<u64>::try_decode(&pt, Encoding::simd()).unwrap();

		// let m2 = Vec::<u64>::try_decode(&sk2_ideal.try_decrypt(&ct2).
		// unwrap(), Encoding::simd()) .unwrap();
		dbg!(m, m2);
	}

	#[test]
	pub fn pcks() {
		let mut rng = thread_rng();
		let params = Arc::new(BfvParameters::default(10, 8));
		let no_of_parties = 4;

		let parties = (0..no_of_parties)
			.map(|_| Party {
				key: SecretKey::random(&params, &mut rng),
				rlk_eph_key: SecretKey::random(&params, &mut rng),
			})
			.collect_vec();
		let cpk = gen_ckg(&params, &parties);

		let m = params.plaintext.random_vec(params.degree(), &mut rng);
		let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
		let ct = cpk.try_encrypt(&pt, &mut rng).unwrap();

		let to_sk = SecretKey::random(&params, &mut rng);
		let to_pk = PublicKey::new(&to_sk, &mut rng);
		let pcks = Pcks::new(&ct, &to_pk);

		// Public collective key switching
		let shares = parties.iter().map(|p| pcks.gen_share(&p.key)).collect_vec();
		let ct2 = pcks.key_switch(&shares);

		// try decrypting ct2 using to_sk
		let pt = to_sk.try_decrypt(&ct2).unwrap();
		let m2 = Vec::<u64>::try_decode(&pt, Encoding::simd()).unwrap();
		assert_eq!(m, m2);
	}
}
