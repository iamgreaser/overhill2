/*
** OverHill2: An obscenely fast Silent Hill 2 RNG seed grinder
** Version 1.0.0
**
** Copyright (c) GreaseMonkey, 2019
**
** This software is provided 'as-is', without any express or implied
** warranty. In no event will the authors be held liable for any damages
** arising from the use of this software.
**
** Permission is granted to anyone to use this software for any purpose,
** including commercial applications, and to alter it and redistribute it
** freely, subject to the following restrictions:
**
**     1. The origin of this software must not be misrepresented; you
**        must not claim that you wrote the original software. If you use
**        this software in a product, an acknowledgment in the product
**        documentation would be appreciated but is not required.
**
**     2. Altered source versions must be plainly marked as such, and
**        must not be misrepresented as being the original software.
**
**     3. This notice may not be removed or altered from any source
**        distribution.
**
** Special thanks to sh2_luck for their research into how the SH2 RNG
** works, for showing it off in the first place, for providing a table
** that the community could make good use of, and for helping me get
** my head around what the randomisation was actually doing internally.
**
** Given a starting clock time and a carbon code, this can produce a
** full set of seeds in under half a second on my i5-6500.
**
** You will need SSE4.1 to compile this.
** For extra performance, enable AVX purely so you can get VEX opcodes.
** (The 256-bit stuff AVX adds is actually slower by this point.)
*/
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <thread>

#define OVERHILL2_VERSION "1.0.0"

// SSE
#include <xmmintrin.h>
// SSE2
#include <emmintrin.h>
// SSE3
#include <pmmintrin.h>
// SSSE3
#include <tmmintrin.h>
// SSE4.1
#include <smmintrin.h>
#ifdef __AVX2__
// AVX2
#include <immintrin.h>
#endif
// POPCNT
#include <popcntintrin.h>

// First seed = 0x6A4F8C55
// this happens on the 428th call to rand()

using std::thread;

const char briefcase_words[][5] = {
	"open", "damn", "hell", "town",
	"dark", "mama", "down", "love",
	"lock", "mist", "luck", "lose",
	"dose", "over", "dust", "time",
	"help", "kill", "null", "cock",
};

int32_t target_clock_angle = -1;
int32_t target_blood_code = -1;
int32_t target_carbon_code = -1;
int32_t target_spin_code = -1;
int32_t target_bug_code = -1;
int32_t target_arsonist_pos = -1;
int32_t target_briefcase_word_idx = -1;

int32_t thread_count = 8;

//constexpr uint32_t istep = 0x1;
constexpr uint32_t istep = 0x10000000;
//constexpr uint32_t istep = 0x01000000;

const uint32_t rand_pow_multab[31] = {
	0x41C64E6D, 0x42A29A69, 0x6E067F11, 0x4FDDDF21,
	0x5F748241, 0x0B2E1481, 0x76006901, 0x1711D201,
	0x3E67A401, 0x5DDF4801, 0x3FFE9001, 0x10FD2001,
	0x65FA4001, 0x5BF48001, 0x77E90001, 0x6FD20001,
	0x5FA40001, 0x3F480001, 0x7E900001, 0x7D200001,
	0x7A400001, 0x74800001, 0x69000001, 0x52000001,
	0x24000001, 0x48000001, 0x10000001, 0x20000001,
	0x40000001, 0x00000001, 0x00000001
};

const uint32_t rand_pow_addtab[31] = {
	0x00003039, 0x53DC167E, 0x56651C2C, 0x4D1DCF18,
	0x65136930, 0x642B7E60, 0x1935ACC0, 0x36461980,
	0x1EF73300, 0x1F9A6600, 0x05E4CC00, 0x26899800,
	0x38133000, 0x1C266000, 0x684CC000, 0x10998000,
	0x21330000, 0x42660000, 0x04CC0000, 0x09980000,
	0x13300000, 0x26600000, 0x4CC00000, 0x19800000,
	0x33000000, 0x66000000, 0x4C000000, 0x18000000,
	0x30000000, 0x60000000, 0x40000000
};

inline uint32_t powrand_from_seed(uint32_t seed, uint32_t amount)
{
	for ( size_t i = 0 ; i < 31 ; i++ ) {
		if ( (amount & (1<<i)) != 0 ) {
			seed *= rand_pow_multab[i];
			seed += rand_pow_addtab[i];
			seed &= 0x7FFFFFFF;
		}
	}

	return seed;
}

struct ssevec {
	__m128i v;

	constexpr ssevec(__m128i v)
		: v(v)
	{
	}

	ssevec operator +(ssevec b) {
		return ssevec(_mm_add_epi32(this->v, b.v));
	}

	ssevec operator *(ssevec b) {
		return ssevec(_mm_mullo_epi32(this->v, b.v));
	}

	ssevec operator +(uint32_t b) {
		return ssevec(_mm_add_epi32(this->v, _mm_set1_epi32(b)));
	}

	ssevec operator *(uint32_t b) {
		return ssevec(_mm_mullo_epi32(this->v, _mm_set1_epi32(b)));
	}

	ssevec operator %(ssevec b) {
		// TODO SIMD version
		return ssevec(_mm_setr_epi32(
			_mm_extract_epi32(this->v, 0) % _mm_extract_epi32(b.v, 0),
			_mm_extract_epi32(this->v, 1) % _mm_extract_epi32(b.v, 1),
			_mm_extract_epi32(this->v, 2) % _mm_extract_epi32(b.v, 2),
			_mm_extract_epi32(this->v, 3) % _mm_extract_epi32(b.v, 3)));
	}

	ssevec operator %(uint32_t b) {
		// TODO SIMD version
		return ssevec(_mm_setr_epi32(
			_mm_extract_epi32(this->v, 0) % b,
			_mm_extract_epi32(this->v, 1) % b,
			_mm_extract_epi32(this->v, 2) % b,
			_mm_extract_epi32(this->v, 3) % b));
	}

	ssevec operator &(ssevec b) {
		return ssevec(_mm_and_si128(this->v, b.v));
	}

	ssevec operator &(uint32_t b) {
		return ssevec(_mm_and_si128(this->v, _mm_set1_epi32(b)));
	}
};
constexpr ssevec _ZERO_I32V4((__m128i){});

#ifdef __AVX2__
struct avxvec {
	__m256i v;

	constexpr avxvec(__m256i v)
		: v(v)
	{
	}

	avxvec operator +(avxvec b) {
		return avxvec(_mm256_add_epi32(this->v, b.v));
	}

	avxvec operator *(avxvec b) {
		return avxvec(_mm256_mullo_epi32(this->v, b.v));
	}

	avxvec operator +(uint32_t b) {
		return avxvec(_mm256_add_epi32(this->v, _mm256_set1_epi32(b)));
	}

	avxvec operator *(uint32_t b) {
		return avxvec(_mm256_mullo_epi32(this->v, _mm256_set1_epi32(b)));
	}

	avxvec operator %(avxvec b) {
		// TODO SIMD version
		return avxvec(_mm256_setr_epi32(
			_mm256_extract_epi32(this->v, 0) % _mm256_extract_epi32(b.v, 0),
			_mm256_extract_epi32(this->v, 1) % _mm256_extract_epi32(b.v, 1),
			_mm256_extract_epi32(this->v, 2) % _mm256_extract_epi32(b.v, 2),
			_mm256_extract_epi32(this->v, 3) % _mm256_extract_epi32(b.v, 3),
			_mm256_extract_epi32(this->v, 4) % _mm256_extract_epi32(b.v, 4),
			_mm256_extract_epi32(this->v, 5) % _mm256_extract_epi32(b.v, 5),
			_mm256_extract_epi32(this->v, 6) % _mm256_extract_epi32(b.v, 6),
			_mm256_extract_epi32(this->v, 7) % _mm256_extract_epi32(b.v, 7)));
	}

	avxvec operator %(uint32_t b) {
		// TODO SIMD version
		return avxvec(_mm256_setr_epi32(
			_mm256_extract_epi32(this->v, 0) % b,
			_mm256_extract_epi32(this->v, 1) % b,
			_mm256_extract_epi32(this->v, 2) % b,
			_mm256_extract_epi32(this->v, 3) % b,
			_mm256_extract_epi32(this->v, 4) % b,
			_mm256_extract_epi32(this->v, 5) % b,
			_mm256_extract_epi32(this->v, 6) % b,
			_mm256_extract_epi32(this->v, 7) % b));
	}

	avxvec operator &(avxvec b) {
		return avxvec(_mm256_and_si256(this->v, b.v));
	}

	avxvec operator &(uint32_t b) {
		return avxvec(_mm256_and_si256(this->v, _mm256_set1_epi32(b)));
	}
};
constexpr avxvec _ZERO_I32V8((__m256i){});
#endif

template <typename T> inline constexpr T zero(void);
template <> inline constexpr uint32_t zero(void) { return 0; }
template <> inline constexpr ssevec zero(void) { return _ZERO_I32V4; }
#ifdef __AVX2__
template <> inline constexpr avxvec zero(void) { return _ZERO_I32V8; }
#endif

template <typename T> inline constexpr uint32_t simd_width(void);
template <> inline constexpr uint32_t simd_width<uint32_t>(void) { return 1; }
template <> inline constexpr uint32_t simd_width<ssevec>(void) { return 4; }
#ifdef __AVX2__
template <> inline constexpr uint32_t simd_width<avxvec>(void) { return 8; }
#endif

template <typename T> inline constexpr T simd_count_ascending(void);
template <> inline constexpr uint32_t simd_count_ascending<uint32_t>(void) { return 0; }
template <> inline constexpr ssevec simd_count_ascending<ssevec>(void) {
	return ssevec((__m128i){
		(1ULL<<32ULL)|0ULL,
		(3ULL<<32ULL)|2ULL,
	});
}
#ifdef __AVX2__
template <> inline constexpr avxvec simd_count_ascending<avxvec>(void) {
	return avxvec((__m256i){
		(1ULL<<32ULL)|0ULL,
		(3ULL<<32ULL)|2ULL,
		(5ULL<<32ULL)|4ULL,
		(7ULL<<32ULL)|6ULL,
	});
}
#endif

template <typename T> inline constexpr uint32_t open_popmask(void);
template <> inline constexpr uint32_t open_popmask<uint32_t>(void) { return 1; }
template <> inline constexpr uint32_t open_popmask<ssevec>(void) { return (1<<4)-1; }
#ifdef __AVX2__
template <> inline constexpr uint32_t open_popmask<avxvec>(void) { return (1<<8)-1; }
#endif

template <int N> inline constexpr uint32_t _rand_coeff_mul(void);
template <> inline constexpr uint32_t _rand_coeff_mul<0>(void) {
	return 1;
}
template <int N> inline constexpr uint32_t _rand_coeff_mul(void) {
	return (_rand_coeff_mul<N-1>() * 1103515245) & 0x7FFFFFFF;
}

template <int N> inline constexpr uint32_t _rand_coeff_add(void);
template <> inline constexpr uint32_t _rand_coeff_add<0>(void) {
	return 0;
}
template <int N> inline constexpr uint32_t _rand_coeff_add(void) {
	return (_rand_coeff_add<N-1>() + _rand_coeff_mul<N-1>()*12345) & 0x7FFFFFFF;
}

template <int N, typename T> inline T randn(T seed) {
	return (seed * _rand_coeff_mul<N>() + _rand_coeff_add<N>()) & 0x7FFFFFFF;
}

template <typename T> inline T if_greater_else_zero(T base, uint32_t cmp, uint32_t if_true);
template <> inline uint32_t if_greater_else_zero(uint32_t base, uint32_t cmp, uint32_t if_true) {
	return (base > cmp)*if_true;
}
template <> inline ssevec if_greater_else_zero(ssevec base, uint32_t cmp, uint32_t if_true) {
	return ssevec(_mm_cmpgt_epi32(base.v, _mm_set1_epi32(cmp))) & if_true;
}
#ifdef __AVX2__
template <> inline avxvec if_greater_else_zero(avxvec base, uint32_t cmp, uint32_t if_true) {
	return avxvec(_mm256_cmpgt_epi32(base.v, _mm256_set1_epi32(cmp))) & if_true;
}
#endif

template <typename T, typename V> inline T if_equal_else_zero(T base, uint32_t cmp, V if_true);
template <> inline uint32_t if_equal_else_zero(uint32_t base, uint32_t cmp, uint32_t if_true) {
	return (base == cmp)*if_true;
}
template <> inline ssevec if_equal_else_zero(ssevec base, uint32_t cmp, uint32_t if_true) {
	return ssevec(_mm_cmpeq_epi32(base.v, _mm_set1_epi32(cmp))) & if_true;
}
template <> inline ssevec if_equal_else_zero(ssevec base, uint32_t cmp, ssevec if_true) {
	return ssevec(_mm_cmpeq_epi32(base.v, _mm_set1_epi32(cmp))) & if_true.v;
}
#ifdef __AVX2__
template <> inline avxvec if_equal_else_zero(avxvec base, uint32_t cmp, uint32_t if_true) {
	return avxvec(_mm256_cmpeq_epi32(base.v, _mm256_set1_epi32(cmp))) & if_true;
}
template <> inline avxvec if_equal_else_zero(avxvec base, uint32_t cmp, avxvec if_true) {
	return avxvec(_mm256_cmpeq_epi32(base.v, _mm256_set1_epi32(cmp))) & if_true.v;
}
#endif

template <typename T, typename U> inline T vec_max(T a, U b);
template <> inline uint32_t vec_max(uint32_t a, uint32_t b) {
	return (a > b ? a : b);
}
template <> inline ssevec vec_max(ssevec a, ssevec b) {
	return ssevec(_mm_max_epu32(a.v, b.v));
}
#ifdef __AVX2__
template <> inline avxvec vec_max(avxvec a, avxvec b) {
	return avxvec(_mm256_max_epu32(a.v, b.v));
}
#endif

template <typename T, typename U> inline T if_lessequal_else_zero(T base, U cmp, uint32_t if_true);
template <> inline uint32_t if_lessequal_else_zero(uint32_t base, uint32_t cmp, uint32_t if_true) {
	return (base <= cmp)*if_true;
}
template <> inline ssevec if_lessequal_else_zero(ssevec base, uint32_t cmp, uint32_t if_true) {
	return ssevec(_mm_xor_si128(_mm_cmpgt_epi32(base.v, _mm_set1_epi32(cmp)), _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()))) & if_true;
}
template <> inline ssevec if_lessequal_else_zero(ssevec base, ssevec cmp, uint32_t if_true) {
	return ssevec(_mm_xor_si128(_mm_cmpgt_epi32(base.v, cmp.v), _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()))) & if_true;
}
#ifdef __AVX2__
template <> inline avxvec if_lessequal_else_zero(avxvec base, uint32_t cmp, uint32_t if_true) {
	return avxvec(_mm256_xor_si256(_mm256_cmpgt_epi32(base.v, _mm256_set1_epi32(cmp)), _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()))) & if_true;
}
template <> inline avxvec if_lessequal_else_zero(avxvec base, avxvec cmp, uint32_t if_true) {
	return avxvec(_mm256_xor_si256(_mm256_cmpgt_epi32(base.v, cmp.v), _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()))) & if_true;
}
#endif

// This was going to be a <typename T, int Step> function,
// but C++ decided to be a piece of shit.
template <typename T> T simd_rand_ascending(uint32_t seed, uint32_t offs, uint32_t step);
template <> uint32_t simd_rand_ascending(uint32_t seed, uint32_t offs, uint32_t step) {
	return powrand_from_seed(seed, offs);
}
template <> ssevec simd_rand_ascending(uint32_t seed, uint32_t offs, uint32_t step) {
	return ssevec(_mm_setr_epi32(
		powrand_from_seed(seed, offs+0*step),
		powrand_from_seed(seed, offs+1*step),
		powrand_from_seed(seed, offs+2*step),
		powrand_from_seed(seed, offs+3*step)));
}
#ifdef __AVX2__
template <> avxvec simd_rand_ascending(uint32_t seed, uint32_t offs, uint32_t step) {
	return avxvec(_mm256_setr_epi32(
		powrand_from_seed(seed, offs+0*step),
		powrand_from_seed(seed, offs+1*step),
		powrand_from_seed(seed, offs+2*step),
		powrand_from_seed(seed, offs+3*step),
		powrand_from_seed(seed, offs+4*step),
		powrand_from_seed(seed, offs+5*step),
		powrand_from_seed(seed, offs+6*step),
		powrand_from_seed(seed, offs+7*step)));
}
#endif

template <typename T> inline uint32_t simd_elem(T vec, uint32_t e);
template <> inline uint32_t simd_elem(uint32_t vec, uint32_t e) { return vec; }
template <> inline uint32_t simd_elem(ssevec vec, uint32_t e) {
	switch(e&3) {
		default:
		case 0: return _mm_extract_epi32(vec.v, 0);
		case 1: return _mm_extract_epi32(vec.v, 1);
		case 2: return _mm_extract_epi32(vec.v, 2);
		case 3: return _mm_extract_epi32(vec.v, 3);
	}
}
#ifdef __AVX2__
template <> inline uint32_t simd_elem(avxvec vec, uint32_t e) {
	switch(e&7) {
		default:
		case 0: return _mm256_extract_epi32(vec.v, 0);
		case 1: return _mm256_extract_epi32(vec.v, 1);
		case 2: return _mm256_extract_epi32(vec.v, 2);
		case 3: return _mm256_extract_epi32(vec.v, 3);
		case 4: return _mm256_extract_epi32(vec.v, 4);
		case 5: return _mm256_extract_epi32(vec.v, 5);
		case 6: return _mm256_extract_epi32(vec.v, 6);
		case 7: return _mm256_extract_epi32(vec.v, 7);
	}
}
#endif

template <typename T> inline uint32_t mask_equal(T base, uint32_t cmp);
template <> inline uint32_t mask_equal(uint32_t base, uint32_t cmp) { return (uint32_t)(base == cmp); }
template <> inline uint32_t mask_equal(ssevec base, uint32_t cmp) {
	return _mm_movemask_ps((__m128)_mm_cmpeq_epi32(base.v, _mm_set1_epi32(cmp)));
}
#ifdef __AVX2__
template <> inline uint32_t mask_equal(avxvec base, uint32_t cmp) {
	return _mm256_movemask_ps((__m256)_mm256_cmpeq_epi32(base.v, _mm256_set1_epi32(cmp)));
}
#endif

template <typename T> class Sim
{
public:
	T m_seed = zero<T>();
	T m_arsonist = zero<T>();
	T m_clock_angle = zero<T>();
	T m_bug_code = zero<T>();
	T m_carbon = zero<T>();
	T m_blood = zero<T>();
	T m_spin = zero<T>();
	T m_briefcase = zero<T>();

public:
	Sim()
	{
	}

	////////////////////////////////////////////////////////////////////

	inline uint32_t fast_solve_all(T seed, uint32_t i)
	{
		uint32_t popmask = open_popmask<T>();

		m_clock_angle = (randn<1>(seed) % 660);
		m_clock_angle = m_clock_angle + if_greater_else_zero(m_clock_angle, 520, 60);

		if ( target_clock_angle >= 0 ) {
			popmask &= mask_equal( m_clock_angle, (uint32_t)target_clock_angle );
			if ( popmask == 0 ) {
				return 0;
			}
		}

		T code0seed = randn<7>(seed);
		m_carbon = (
			(randn<0*3>(code0seed) % 9) * 1000
			+ (randn<1*3>(code0seed) % 9) * 100
			+ (randn<2*3>(code0seed) % 9) * 10
			+ (randn<3*3>(code0seed) % 9) * 1
			+ 1111);

		if ( target_carbon_code >= 0 ) {
			popmask &= mask_equal( m_carbon, (uint32_t)target_carbon_code );
			if ( popmask == 0 ) {
				return 0;
			}
		}


		T code1seed = randn<8>(seed);
		T code1digit0 = (randn<0*3>(code1seed) % 9);
		T code1digit1 = (randn<1*3>(code1seed) % 9);
		T code1digit2 = (randn<2*3>(code1seed) % 9);
		T code1digit3 = (randn<3*3>(code1seed) % 9);

		m_blood = (
			(code1digit0) * 1000
			+ (code1digit1) * 100
			+ (code1digit2) * 10
			+ (code1digit3)
			+ 1111);

		if ( target_blood_code >= 0 ) {
			popmask &= mask_equal( m_blood, (uint32_t)target_blood_code );
			if ( popmask == 0 ) {
				return 0;
			}
		}

		T code2seed = randn<9>(seed);
		T code2digit0 = ((randn<0*3>(code2seed) & (8U-1)) + 1 + code1digit0) % 9;
		T code2digit1 = ((randn<1*3>(code2seed) & (8U-1)) + 1 + code1digit1) % 9;
		T code2digit2 = ((randn<2*3>(code2seed) & (8U-1)) + 1 + code1digit2) % 9;
		T code2digit3 = ((randn<3*3>(code2seed) & (8U-1)) + 1 + code1digit3) % 9;
		m_spin = (
			(code2digit0) * 1000
			+ (code2digit1) * 100
			+ (code2digit2) * 10
			+ (code2digit3)
			+ 1111);

		T digit0 = randn<19>(seed) % 9;
		T digit1 = randn<20>(seed) & (8U-1);
		T digit2 = randn<21>(seed) % 7;
		m_bug_code = (
			(digit0) * 100
			+ (digit1+if_lessequal_else_zero(digit0,digit1,1)) * 10
			+ (digit2+if_lessequal_else_zero(digit0,digit2,1)+if_lessequal_else_zero(digit1,digit2,1))
			+ 111
		);

		m_arsonist = zero<T>();
		T ars_seed = randn<22>(seed);
		T ars6 = if_equal_else_zero(ars_seed % 6U, 5U, 5U);
		ars_seed = randn<1>(ars_seed);
		T ars5 = if_equal_else_zero(ars_seed % 5U, 4U, 4U);
		ars_seed = randn<1>(ars_seed);
		T ars4 = if_equal_else_zero(ars_seed & (4U-1), 3U, 3U);
		ars_seed = randn<1>(ars_seed);
		T ars3 = if_equal_else_zero(ars_seed % 3U, 2U, 2U);
		ars_seed = randn<1>(ars_seed);
		T ars2 = if_equal_else_zero(ars_seed & (2U-1), 1U, 1U);
		m_arsonist = vec_max(
			vec_max(vec_max(ars6, ars5), ars4),
			vec_max(ars3, ars2));

		m_briefcase = randn<30>(seed) % 19;

		if ( true ) {
			//popmask && (popmask &= mask_equal( m_spin, 1234 ));
			//popmask && (popmask &= mask_equal( m_bug_code, 239 ));
			//popmask && (popmask &= mask_equal( m_arsonist, 1-1 ));
			//popmask && (popmask &= mask_equal( m_briefcase, 8 ));
		}

		if ( false && popmask != 0 ) {
			for ( uint32_t j = 0; j < simd_width<T>(); j++ ) {
				if ( (popmask & (1<<j)) != 0 ) {
					printf("%10u,0x%08X,%02d:%02d,%04d,%04d,%04d,%03d,%1d,%2d,%4s\n",
						i+j,
						simd_elem(seed, j),
						simd_elem(m_clock_angle, j) / 60,
						simd_elem(m_clock_angle, j) % 60,

						simd_elem(m_blood, j),
						simd_elem(m_carbon, j),
						simd_elem(m_spin, j),
						simd_elem(m_bug_code, j),
						simd_elem(m_arsonist+1, j),

						simd_elem(m_briefcase, j),
						briefcase_words[simd_elem(m_briefcase, j)]);
				}
			}
		}

		return popmask;
	}
};


template <typename T, int Step> void grind_seed_block(uint32_t offs, uint32_t beg, uint32_t len)
{
	Sim<T> sim;

	// Optimise for the clock case
	// TODO: optimise for some other heavily biased RNG cases
	if ( Step == 1 ) {
		if ( target_clock_angle >= 0 ) {
			uint32_t clockseedoffs = (target_clock_angle+2)&0x3;
			grind_seed_block<T,4>(clockseedoffs, beg, len);
			return;
		}
	}

	uint32_t iseed = powrand_from_seed(0x6A4F8C55, beg);
	T seed = simd_rand_ascending<T>(iseed, offs, Step);
	for ( uint32_t i = beg+offs ; i < beg+len ; i += simd_width<T>()*Step, seed = randn<simd_width<T>()*Step>(seed) ) {
		uint32_t mask = sim.fast_solve_all(seed, i);

		if ( true && mask != 0 ) {
			for ( uint32_t j = 0; j < simd_width<T>(); j++ ) {
				if ( (mask & (1<<j)) != 0 ) {
					printf("%10u,0x%08X,%02d:%02d,%04d,%04d,%04d,%03d,%1d,%2d,%4s\n",
						i+j,
						simd_elem(seed, j),
						simd_elem(sim.m_clock_angle, j) / 60,
						simd_elem(sim.m_clock_angle, j) % 60,

						simd_elem(sim.m_blood, j),
						simd_elem(sim.m_carbon, j),
						simd_elem(sim.m_spin, j),
						simd_elem(sim.m_bug_code, j),
						simd_elem(sim.m_arsonist+1, j),

						simd_elem(sim.m_briefcase, j),
						briefcase_words[simd_elem(sim.m_briefcase, j)]);
				}
			}
		}
	}
}

template <typename T> void grind_whole_space(void)
{
	for ( uint32_t ibase = 0 ; ibase < 0x80000000U ; ibase += istep ) {
		if ( thread_count == 1 ) {
			grind_seed_block<T,1>(0, ibase, istep);

		} else {
			thread *threads[thread_count];

			for ( uint32_t i = 0 ; i < thread_count ; i++ ) {
				size_t block_beg = ibase+(i+0)*istep/thread_count;
				size_t block_end = ibase+(i+1)*istep/thread_count;
				size_t block_len = block_end - block_beg;
				threads[i] = new thread(grind_seed_block<T,1>, 0, block_beg, block_len);
			}
			for ( uint32_t i = 0 ; i < thread_count ; i++ ) {
				threads[i]->join();
				delete threads[i];
			}
		}

		printf("%08X %11u\n", ibase+istep, ibase+istep);
	}

}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	size_t curarg = 1;

	// One of these days I'll learn how you're supposed to do this in C++.
	// In the meantime, here's the C way.
	while ( curarg < argc ) {
		if ( !strcmp(argv[curarg], "--threads") && curarg+2 <= argc ) {
			curarg++;
			char *endptr = argv[curarg];
			thread_count = (uint32_t)strtol(argv[curarg], &endptr, 0);
			assert ( endptr != argv[curarg] && "Please provide a number." );
			assert ( 1 <= thread_count && thread_count < 64 && "If you really, honestly, do need more than 64 threads, recompile this program from source." );
			curarg++;

		} else if ( !strcmp(argv[curarg], "--clock") && curarg+2 <= argc ) {
			curarg++;
			uint32_t c_hours = 13;
			uint32_t c_mins = 61;
			int sscanf_result = sscanf(argv[curarg], "%d:%d", &c_hours, &c_mins);
			assert ( sscanf_result == 2 && "Please provide a valid time in HH:MM format.");

			// Because 12 hour time is *totally* mathematically convenient.
			assert ( c_hours >= 0 && c_hours <= 12 && "Please provide a valid time in HH:MM format.");
			if ( c_hours == 12 ) { c_hours = 0; }
			assert ( c_hours >= 0 && c_hours < 12 && "Please provide a valid time in HH:MM format.");
			assert ( c_mins >= 0 && c_mins < 60 && "Please provide a valid time in HH:MM format.");
			target_clock_angle = c_hours*60+c_mins;
			curarg++;

		} else if ( !strcmp(argv[curarg], "--carbon") && curarg+2 <= argc ) {
			curarg++;
			char *endptr = argv[curarg];
			target_carbon_code = (uint32_t)strtol(argv[curarg], &endptr, 10);
			assert ( endptr != argv[curarg] && "Please provide a number." );
			assert ( 1111 <= target_carbon_code && target_carbon_code < 9999 && "Please provide a valid code." );
			curarg++;

		} else if ( !strcmp(argv[curarg], "--blood") && curarg+2 <= argc ) {
			curarg++;
			char *endptr = argv[curarg];
			target_blood_code = (uint32_t)strtol(argv[curarg], &endptr, 10);
			assert ( endptr != argv[curarg] && "Please provide a number." );
			assert ( 1111 <= target_blood_code && target_blood_code < 9999 && "Please provide a valid code." );
			curarg++;

		} else {
			fprintf(stderr, "OverHill2: An obscenely fast Silent Hill 2 RNG seed grinder\n");
			fprintf(stderr, "Version %s\n", OVERHILL2_VERSION);
			fprintf(stderr, "Copyright (c) 2019, GreaseMonkey\n");
			fprintf(stderr, "See the LICENCE.txt file this came with.\n");
			fprintf(stderr, "(Alternatively, imagine the zlib licence with my name on it.)\n");
			fprintf(stderr, "\n");
			fprintf(stderr, "usage:\n");
			fprintf(stderr, "\t%s [--threads 8] [--clock HH:MM] [--carbon NNNN] [--blood NNNN]\n", argv[0]);
			fprintf(stderr, "\n");
			fprintf(stderr, "arguments:\n");
			fprintf(stderr, "\t--threads N    Specify the number of CPU threads to use.\n");
			fprintf(stderr, "\t--clock HH:MM  Target a specific starting clock value.\n");
			fprintf(stderr, "\t--carbon NNNN  Target a specific carbon code.\n");
			fprintf(stderr, "\t--blood NNNN   Target a specific blood code.\n");
			return 1;
		}
	}

	//grind_whole_space<uint32_t>();
	grind_whole_space<ssevec>();

	// In practice, the AVX2 version is slower than the SSE4.1 version.
	// You get more speed by enabling AVX for the VEX prefix,
	// and then sticking with the SSE4.1 version.
	//grind_whole_space<avxvec>();

	return 0;
}
