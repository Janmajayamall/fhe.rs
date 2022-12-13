#![feature(portable_simd)]
use std::{iter, simd::Simd};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fhe_math::zq::Modulus;
use itertools::{izip, Itertools};
use rand::thread_rng;

pub fn zq_benchmark(c: &mut Criterion) {
	let mut group = c.benchmark_group("zq");
	group.sample_size(50);

	let p = 4611686018326724609;
	let mut rng = thread_rng();

	for vector_size in [1024usize, 4096, 1 << 15].iter() {
		let q = Modulus::new(p).unwrap();
		let mut a = q.random_vec(*vector_size, &mut rng);
		let c = q.random_vec(*vector_size, &mut rng);
		let c_shoup = q.shoup_vec(&c);
		let scalar = c[0];

		group.bench_function(BenchmarkId::new("add_vec", vector_size), |b| {
			b.iter(|| q.add_vec(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("add_vec_vt", vector_size), |b| unsafe {
			b.iter(|| q.add_vec_vt(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("add_vec_simd", vector_size), |b| {
			b.iter(|| q.add_vec_simd(&mut a, &c))
		});

		group.bench_function(BenchmarkId::new("add_vec2_simd", vector_size), |b| {
			let (a0, a1, a2) = a.as_simd_mut::<8>();
			let (c0, c1, c2) = c.as_simd::<8>();
			assert!(a0.len() + a2.len() == 0);

			b.iter(|| {
				q.add_simd_vec(a1, c1);
			});
		});

		group.bench_function(BenchmarkId::new("sub_vec", vector_size), |b| {
			b.iter(|| q.sub_vec(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("sub_vec_simd", vector_size), |b| {
			b.iter(|| q.sub_vec_simd(&mut a, &c))
		});

		group.bench_function(BenchmarkId::new("neg_vec", vector_size), |b| {
			b.iter(|| q.neg_vec(&mut a));
		});

		group.bench_function(BenchmarkId::new("mul_vec", vector_size), |b| {
			b.iter(|| q.mul_vec(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("mul_vec_vt", vector_size), |b| unsafe {
			b.iter(|| q.mul_vec_vt(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("mul_shoup_vec", vector_size), |b| {
			b.iter(|| q.mul_shoup_vec(&mut a, &c, &c_shoup));
		});

		group.bench_function(BenchmarkId::new("scalar_mul_vec", vector_size), |b| {
			b.iter(|| q.scalar_mul_vec(&mut a, scalar));
		});

		group.bench_function(BenchmarkId::new("mul_vec_simd", vector_size), |b| {
			let (a0, a1, a2) = a.as_simd_mut::<8>();
			let (c0, c1, c2) = c.as_simd::<8>();

			b.iter(|| q.mul_simd_vec(a1, c1));
		});

		let ac = izip!(a.clone(), c.clone())
			.map(|(_a, _c)| _a as u128 * _c as u128)
			.collect_vec();
		group.bench_function(BenchmarkId::new("reduce_opt_u128_simd", vector_size), |b| {
			let low_mask = (1u128 << 64) - 1;
			let mut ac2: (Vec<u64>, Vec<u64>) = ac
				.iter()
				.map(|v| ((v >> 64) as u64, (v & low_mask) as u64))
				.unzip();
			let (a_hi0, a_hi1, a_hi2) = ac2.0.as_simd_mut::<8>();
			let (a_lo0, a_lo1, a_lo2) = ac2.1.as_simd_mut::<8>();
			assert!(a_hi0.len() + a_hi2.len() == 0);

			b.iter(|| {
				q.reduce_opt_u128_simd_vec(a_hi1, a_lo1);
			});
		});

		group.bench_function(
			BenchmarkId::new(
				"
		reduce_opt_u128",
				vector_size,
			),
			|b| {
				let ac = ac.clone();
				let mut res = vec![0u64; ac.len()];
				b.iter(|| {
					for (v, r) in ac.iter().zip(res.iter_mut()) {
						*r = q.reduce_opt_u128(*v);
					}
				});
			},
		);

		group.bench_function(BenchmarkId::new("mul_hi", vector_size), |b| {
			let (a_hi0, a_hi1, a_hi2) = a.as_simd_mut::<8>();
			let (_, c_hi1, _) = c.as_simd::<8>();
			b.iter(|| {
				q.mulhi_simd_vec(a_hi1, c_hi1 );
			});
		});
	}

	group.finish();
}

criterion_group!(zq, zq_benchmark);
criterion_main!(zq);
