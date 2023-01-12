use fhe::bfv::{
	BfvParameters, BfvParametersBuilder, Ciphertext, Ckg, Encoding, EvaluationKey, GaloisKey, Pcks,
	Plaintext, PublicKey, RelinearizationKey, Rkg, Rtg, SecretKey,
};
use fhe_math::{
	rq::{Context, SubstitutionExponent},
	zq::Modulus,
};
use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
use itertools::{izip, Itertools};
use rand::thread_rng;
use std::{collections::HashMap, error::Error, sync::Arc};

struct Party {
	key: SecretKey,
	rlk_eph_key: SecretKey,
}

fn evaluation_key(params: &Arc<BfvParameters>, parties: &[Party]) -> EvaluationKey {
	// gen rotation keys for inner product
	let mut g_elems = vec![];
	let mut i = 1;
	// Row rotation
	g_elems.push(params.degree() * 2 - 1);
	// Column rotations
	let q = Modulus::new(2 * params.degree() as u64).unwrap();
	while i < params.degree() / 2 {
		g_elems.push(q.pow(3, i as u64) as usize);
		i *= 2;
	}

	let ctx = Arc::new(Context::new(params.moduli(), params.degree()).unwrap());
	let mut gk = HashMap::new();
	g_elems.iter().for_each(|g| {
		let crps = Rtg::sample_crps(&params, 0, 0);
		let rtg = Rtg::new(
			params,
			0,
			0,
			&crps,
			SubstitutionExponent::new(&ctx, *g).unwrap(),
		);
		let shares = parties.iter().map(|p| rtg.gen_share(&p.key)).collect_vec();
		let agg_shares = rtg.aggregate_shares(&shares);
		gk.insert(*g, GaloisKey::new_from_rtg(&rtg, &agg_shares));
	});
	EvaluationKey::new_from_gks(&params, 0, 0, gk, true).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
	let params = Arc::new(
		BfvParametersBuilder::new()
			.set_degree(128)
			.set_plaintext_modulus(65537)
			.set_moduli_sizes(&[60, 60, 60])
			.build()?,
	);
	let mut rng = thread_rng();

	// no. of parties
	let n = 10;

	let parties = (0..n)
		.map(|_| Party {
			key: SecretKey::random(&params, &mut rng),
			rlk_eph_key: SecretKey::random(&params, &mut rng),
		})
		.collect_vec();

	// Collective key generation for pk
	// TODO: supply rng as params
	let crp = Ckg::sample_crp(&params);
	let ckg = Ckg::new(&params, &crp);
	let shares = parties.iter().map(|p| ckg.gen_share(&p.key)).collect_vec();
	let agg_shares = ckg.aggregate_shares(&shares);
	let pk = PublicKey::new_from_ckg(&ckg, &agg_shares);

	// gen rlk keys
	let crps = Rkg::sample_crps(&params, 0, 0);
	let rkg = Rkg::new(&params, 0, 0, &crps);
	let shares1 = parties
		.iter()
		.map(|p| rkg.gen_share_round1(&params, &p.key, &p.rlk_eph_key))
		.collect_vec();
	let agg_shares1 = rkg.aggregate_shares(&shares1);
	let shares2 = parties
		.iter()
		.map(|p| rkg.gen_share_round2(&params, &agg_shares1, &p.key, &p.rlk_eph_key))
		.collect_vec();
	let agg_shares2 = rkg.aggregate_shares(&shares2);
	let rlk = RelinearizationKey::new_from_rkg(&rkg, &agg_shares2, &agg_shares1);

	// evluation key
	let eval_key = evaluation_key(&params, &parties);

	// gen data
	let q = Modulus::new(params.plaintext()).unwrap();
	let data = parties
		.iter()
		.map(|p| {
			let m = q.random_vec(params.degree(), &mut rng);
			let pt = Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap();
			let ct = pk.try_encrypt(&pt, &mut rng).unwrap();
			(m, pt, ct)
		})
		.collect_vec();

	let ct_data = data.iter().map(|d| d.2.clone()).collect_vec();

	// query
	let user_sk = SecretKey::random(&params, &mut rng);
	let user_pk = PublicKey::new(&user_sk, &mut rng);
	let query_index = 5;
	let query_m = (0..parties.len())
		.map(|i| (i == query_index) as u64)
		.collect_vec();
	let query_pt = Plaintext::try_encode(&query_m, Encoding::simd(), &params).unwrap();
	let query_ct = pk.try_encrypt(&query_pt, &mut rng).unwrap();

	// process
	let filter_pts = (0..parties.len())
		.map(|i| {
			let mut m = vec![0; parties.len()];
			m[i] = 1 as u64;
			Plaintext::try_encode(&m, Encoding::simd(), &params).unwrap()
		})
		.collect_vec();

	let expanded_indexes = filter_pts
		.iter()
		.map(|pt| {
			let res = &query_ct * pt;
			eval_key.computes_inner_sum(&res).unwrap()
		})
		.collect_vec();

	let mut result = Ciphertext::zero(&params);
	izip!(ct_data.iter(), expanded_indexes.iter()).for_each(|(d, i)| {
		let mut r = d * i;
		rlk.relinearizes(&mut r).unwrap();
		result += &r;
	});

	// key switch to user sk
	let pcks = Pcks::new(&result, &user_pk);
	let shares = parties.iter().map(|p| pcks.gen_share(&p.key)).collect_vec();
	let user_result = pcks.key_switch(&shares);

	let pt = user_sk.try_decrypt(&user_result).unwrap();
	let m = Vec::<u64>::try_decode(&pt, Encoding::simd()).unwrap();

	assert_eq!(&m, &data[query_index].0);

	Ok(())
}
