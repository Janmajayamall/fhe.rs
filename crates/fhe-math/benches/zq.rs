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
			b.iter(|| q.add_vec_simd::<8>(&mut a, &c))
		});

		let (a0, a1, a2) = a.as_simd::<8>();
		let (c0, c1, c2) = c.as_simd::<8>();
		assert!(a0.len() + a2.len() == 0);
		group.bench_function(BenchmarkId::new("add_simd", vector_size), |b| {
			b.iter(|| {
				izip!(a1, c1).for_each(|(a, c)| {
					q.add_simd::<8>(*a, *c);
				});
			})
		});

		group.bench_function(BenchmarkId::new("sub_vec", vector_size), |b| {
			b.iter(|| q.sub_vec(&mut a, &c));
		});

		group.bench_function(BenchmarkId::new("sub_vec_simd", vector_size), |b| {
			b.iter(|| q.sub_vec_simd::<8>(&mut a, &c))
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

		let low_mask = (1u128 << 64) - 1;
		let a_sq: (Vec<u64>, Vec<u64>) = a
			.iter()
			.map(|v| {
				let sq = *v as u128 * *v as u128;
				((sq >> 64) as u64, (sq & low_mask) as u64)
			})
			.unzip();
		let (a_hi0, a_hi1, a_hi2) = a_sq.0.as_simd::<8>();
		let (a_lo0, a_lo1, a_lo2) = a_sq.1.as_simd::<8>();
		let barret_lo = Simd::from_array([q.barrett_lo; 8]);
		let low_mask = Simd::from_array([4294967295u64; 8]);
		let shift_32 = Simd::from_array([32u64; 8]);
		let left_shift = Simd::from_array([q.leading_zeros as u64; 8]);
		group.bench_function(BenchmarkId::new("reduce_opt_u128_simd", vector_size), |b| {
			b.iter(|| {
				izip!(a_hi1, a_lo1).for_each(|(a, b)| {
					q.reduce_opt_u128_simd(*a, *b, barret_lo, low_mask, shift_32, left_shift);
				})
			});
		});

		// temporary function to compare performance of reduce_opt_u128 against simd
		let a_sq = a.iter().map(|v| *v as u128 * *v as u128).collect_vec();
		group.bench_function(BenchmarkId::new("reduce_opt_u128", vector_size), |b| {
			b.iter(|| {
				a_sq.iter().for_each(|v| {
					q.reduce_opt_u128(*v);
				})
			});
		});
	}

	group.finish();
}

criterion_group!(zq, zq_benchmark);
criterion_main!(zq);
