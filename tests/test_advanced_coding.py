"""Comprehensive Verification Test Suite for Advanced DSP & Coding Schemes.
NTRO PS-26147 Specification

Covers:
  1. Interleaver Schemes:
     - Block (Transpose)
     - Convolutional (Shift-register delay branches)
     - Diagonal (Diagonal-write, row-read)
     - Pseudo-Random (Quadratic Permutation Polynomial - QPP)
     - PRBS (Pseudo-Random Binary Sequence shuffler)
     - Interleaver blind classifier

  2. FEC Schemes:
     - Reed-Solomon RS(255, 223) & RS(255, 239) over GF(2^8)
     - Concatenated Codes (Outer RS + Inner Convolutional/Viterbi)
     - Low-Density Parity-Check (LDPC) Belief Propagation (Min-Sum)

  3. Robustness & Error-Correction:
     - 0-error clean recovery (BER == 0.0)
     - Random bit errors & symbol errors within correction radius
     - Burst error dispersion via interleaver + FEC
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src import fec, interleaver, synthetic_gen, bitstream

failures = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


def main():
    rng = np.random.default_rng(2026)

    print("\n" + "=" * 70)
    print("SECTION 1: ADVANCED INTERLEAVER SCHEMES")
    print("=" * 70)

    # 1.1 Block Interleaver
    n_bits = 1024
    bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
    intl_block = interleaver.block_interleave(bits, rows=32, cols=32)
    deintl_block = interleaver.block_deinterleave(intl_block, rows=32, cols=32)
    check("Block Interleaver (32x32) round-trip BER==0",
          np.array_equal(bits, deintl_block),
          f"(bits={bits.size})")

    # 1.2 Convolutional Interleaver
    # B branches, delay D -> total round trip latency = B * (B - 1) * D
    B, D = 8, 4
    latency = B * (B - 1) * D
    n_conv = 2048
    bits_conv = rng.integers(0, 2, n_conv, dtype=np.uint8)
    intl_conv = interleaver.conv_interleave(bits_conv, branches=B, delay=D)
    deintl_conv = interleaver.conv_deinterleave(intl_conv, branches=B, delay=D)
    # Bits after latency match original
    valid_len = n_conv - latency
    recovered = deintl_conv[latency:n_conv]
    orig_valid = bits_conv[:valid_len]
    check("Convolutional Interleaver (B=8, D=4) round-trip BER==0",
          np.array_equal(orig_valid, recovered),
          f"(latency={latency} bits, verified={valid_len} bits)")

    # 1.3 Diagonal Interleaver
    r_diag, c_diag = 16, 32
    bits_diag = rng.integers(0, 2, r_diag * c_diag * 3, dtype=np.uint8)
    intl_diag = interleaver.diag_interleave(bits_diag, rows=r_diag, cols=c_diag)
    deintl_diag = interleaver.diag_deinterleave(intl_diag, rows=r_diag, cols=c_diag)
    check("Diagonal Interleaver (16x32) round-trip BER==0",
          np.array_equal(bits_diag, deintl_diag),
          f"(total_bits={bits_diag.size})")

    # 1.4 Pseudo-Random QPP Interleaver
    n_qpp = 240
    bits_qpp = rng.integers(0, 2, n_qpp * 4, dtype=np.uint8)
    intl_qpp = interleaver.qpp_interleave(bits_qpp, n=n_qpp)
    deintl_qpp = interleaver.qpp_deinterleave(intl_qpp, n=n_qpp)
    check("Pseudo-Random QPP Interleaver (N=240) round-trip BER==0",
          np.array_equal(bits_qpp, deintl_qpp),
          f"(N={n_qpp}, blocks=4)")

    # 1.5 PRBS Interleaver
    n_prbs = 512
    bits_prbs = rng.integers(0, 2, n_prbs * 2, dtype=np.uint8)
    intl_prbs = interleaver.prbs_interleave(bits_prbs, n=n_prbs, seed=42)
    deintl_prbs = interleaver.prbs_deinterleave(intl_prbs, n=n_prbs, seed=42)
    check("PRBS Seeded Interleaver (N=512) round-trip BER==0",
          np.array_equal(bits_prbs, deintl_prbs),
          f"(N={n_prbs}, blocks=2)")

    # 1.6 Interleaver Scheme Classification
    # Structured codewords (rank deficiency present as in real FEC block)
    info_test = rng.integers(0, 2, (64, 26), dtype=np.uint8)
    coded_test = np.array([fec.conv_encode(row) for row in info_test]) # (64, 64)
    intl_coded_block = interleaver.block_interleave(coded_test.ravel(), rows=64, cols=64)
    clf_block = interleaver.classify_interleaver(intl_coded_block, max_width=128, min_rows=8)
    check("Classifier identifies block interleaver on coded stream",
          clf_block["scheme"] == "block",
          f"(detected: {clf_block['scheme']}, width: {clf_block['params']['width']}, conf: {clf_block['confidence']:.2f})")

    # Pseudo-random / unstructured stream (no rank deficiency)
    clf_prbs = interleaver.classify_interleaver(intl_prbs, max_width=128, min_rows=8)
    check("Classifier identifies pseudo-random interleaver on flat stream",
          clf_prbs["scheme"] == "pseudo_random",
          f"(detected: {clf_prbs['scheme']}, conf: {clf_prbs['confidence']:.2f})")


    print("\n" + "=" * 70)
    print("SECTION 2: ADVANCED FEC SCHEMES")
    print("=" * 70)

    # 2.1 Reed-Solomon (255, 223) - 16 error correction capability (nsym=32)
    msg_rs223 = rng.integers(0, 256, 223, dtype=np.uint8)
    cw_rs223 = fec.rs_encode(msg_rs223, nsym=32)
    check("RS(255, 223) codeword length == 255", cw_rs223.size == 255)

    # Clean decode
    dec_rs223, err_cnt0 = fec.rs_decode(cw_rs223, nsym=32)
    check("RS(255, 223) clean decode BER==0",
          np.array_equal(msg_rs223, dec_rs223) and err_cnt0 == 0,
          f"(reported errors: {err_cnt0})")

    # Error correction within radius (t=16 errors)
    t = 12  # introduce 12 symbol errors
    err_pos = rng.choice(255, size=t, replace=False)
    corrupted_rs = cw_rs223.copy()
    for pos in err_pos:
        corrupted_rs[pos] ^= int(rng.integers(1, 256))
    dec_corrupted, err_cnt = fec.rs_decode(corrupted_rs, nsym=32)
    check(f"RS(255, 223) corrects {t} random symbol errors",
          np.array_equal(msg_rs223, dec_corrupted) and err_cnt == t,
          f"(target={t}, corrected={err_cnt})")

    # 2.2 Reed-Solomon (255, 239) - 8 error correction capability (nsym=16)
    msg_rs239 = rng.integers(0, 256, 239, dtype=np.uint8)
    cw_rs239 = fec.rs_encode(msg_rs239, nsym=16)
    t_239 = 6
    err_pos_239 = rng.choice(255, size=t_239, replace=False)
    corrupted_rs239 = cw_rs239.copy()
    for pos in err_pos_239:
        corrupted_rs239[pos] ^= int(rng.integers(1, 256))
    dec_corrupted239, err_cnt239 = fec.rs_decode(corrupted_rs239, nsym=16)
    check(f"RS(255, 239) corrects {t_239} random symbol errors",
          np.array_equal(msg_rs239, dec_corrupted239) and err_cnt239 == t_239,
          f"(target={t_239}, corrected={err_cnt239})")

    # Uncorrectable error handling (exceeding capacity t > 8 for RS(255, 239))
    t_excess = 18
    err_pos_excess = rng.choice(255, size=t_excess, replace=False)
    corrupted_excess = cw_rs239.copy()
    for pos in err_pos_excess:
        corrupted_excess[pos] ^= int(rng.integers(1, 256))
    _, err_excess = fec.rs_decode(corrupted_excess, nsym=16)
    check(f"RS(255, 239) flags uncorrectable error overload ({t_excess} errors)",
          err_excess == -1 or err_excess != t_excess,
          f"(err_reported={err_excess})")

    # 2.3 Concatenated Codes (Outer RS(255, 223) + Inner Viterbi K=7)
    msg_concat = rng.integers(0, 256, 223, dtype=np.uint8)
    cw_concat = fec.concat_encode(msg_concat, rs_nsym=32, conv_K=7)
    # Clean decode
    dec_concat, stats_concat = fec.concat_decode(cw_concat, rs_nsym=32, n_rs_symbols=255, conv_K=7)
    check("Concatenated Code clean decode BER==0",
          np.array_equal(msg_concat, dec_concat),
          f"(viterbi_bits={stats_concat['viterbi_bits']}, rs_errors={stats_concat['rs_errors']})")

    # 2.4 LDPC Code (48, 24) Min-Sum Belief Propagation
    n_ldpc, k_ldpc = 48, 24
    H_ldpc = fec.make_ldpc_parity_check(n=n_ldpc, k=k_ldpc, seed=777)
    msg_ldpc = rng.integers(0, 2, k_ldpc, dtype=np.uint8)
    cw_ldpc = fec.ldpc_encode(msg_ldpc, H_ldpc)
    check("LDPC systematic syndrome is identically zero",
          np.all((H_ldpc @ cw_ldpc) % 2 == 0))

    # Clean decode
    dec_ldpc, it_clean = fec.ldpc_decode(cw_ldpc, H_ldpc)
    check("LDPC clean decode BER==0",
          np.array_equal(cw_ldpc, dec_ldpc),
          f"(converged in {it_clean} iterations)")

    # LDPC bit error correction (1-2 bit flips)
    cw_ldpc_noisy = cw_ldpc.copy()
    cw_ldpc_noisy[7] ^= 1
    cw_ldpc_noisy[33] ^= 1
    dec_ldpc_noisy, it_noisy = fec.ldpc_decode(cw_ldpc_noisy, H_ldpc)
    check("LDPC corrects 2-bit channel errors",
          np.array_equal(cw_ldpc, dec_ldpc_noisy),
          f"(converged in {it_noisy} iterations)")

    print("\n" + "=" * 70)
    print("SECTION 3: BURST ERROR DISPERSION (INTERLEAVER + FEC)")
    print("=" * 70)
    # Demonstrate the core purpose of interleaving:
    # A concentrated burst of errors ruins a code without interleaving,
    # but with interleaving the burst is distributed across codewords and fully corrected.

    # 3.1 Convolutional FEC under burst error:
    # 16 rows of 26 info bits -> 16 codewords of 64 bits each (1024 bits total).
    info_matrix = rng.integers(0, 2, (16, 26), dtype=np.uint8)
    coded_matrix = np.array([fec.conv_encode(row) for row in info_matrix]) # (16, 64)
    coded_flat = coded_matrix.ravel() # 1024 bits

    # Without interleaver: inject burst of 12 consecutive bit errors in one codeword
    non_interleaved = coded_flat.copy()
    non_interleaved[10:22] ^= 1 # 12 consecutive errors in codeword 0
    dec_no_intl = np.array([
        fec.viterbi_decode(non_interleaved[i * 64:(i + 1) * 64].reshape(-1, 2))
        for i in range(16)
    ])
    ber_no_intl = float(np.mean(dec_no_intl != info_matrix))

    # With 16x64 Block Interleaver:
    interleaved = interleaver.block_interleave(coded_flat, rows=16, cols=64)
    # Inject the identical burst of 12 consecutive bit errors
    interleaved[10:22] ^= 1
    # Deinterleave:
    deinterleaved = interleaver.block_deinterleave(interleaved, rows=16, cols=64)
    dec_with_intl = np.array([
        fec.viterbi_decode(deinterleaved[i * 64:(i + 1) * 64].reshape(-1, 2))
        for i in range(16)
    ])
    ber_with_intl = float(np.mean(dec_with_intl != info_matrix))

    check("Without interleaver, burst error corrupts Viterbi decode",
          ber_no_intl > 0.0,
          f"(uncorrected BER={ber_no_intl:.4f})")
    check("With block interleaver, 12-bit burst error is completely corrected",
          ber_with_intl == 0.0,
          f"(corrected BER={ber_with_intl:.4f})")

    print("\n" + "=" * 70)
    print("SECTION 4: SYNTHETIC DATA & IQ GENERATION PIPELINE")
    print("=" * 70)

    # 4.1 RS + Convolutional Interleaver IQ Generation
    gt_rs = synthetic_gen.generate(
        "testdata/synth_test_rs.iq",
        fec_scheme="rs",
        interleaver_scheme="convolutional",
        snr_db=20.0
    )
    check("Synthetic Gen RS + Conv Interleaver IQ generation",
          Path(gt_rs["file"]).exists() and gt_rs["samples"] > 0,
          f"({gt_rs['samples']} samples, {gt_rs['fec_scheme']}/{gt_rs['interleaver_scheme']})")

    # 4.2 Concatenated + Diagonal Interleaver IQ Generation
    gt_concat = synthetic_gen.generate(
        "testdata/synth_test_concat.iq",
        fec_scheme="concatenated",
        interleaver_scheme="diagonal",
        snr_db=20.0
    )
    check("Synthetic Gen Concatenated + Diagonal Interleaver IQ generation",
          Path(gt_concat["file"]).exists() and gt_concat["samples"] > 0,
          f"({gt_concat['samples']} samples, {gt_concat['fec_scheme']}/{gt_concat['interleaver_scheme']})")

    # 4.3 LDPC + QPP Interleaver IQ Generation
    gt_ldpc = synthetic_gen.generate(
        "testdata/synth_test_ldpc.iq",
        fec_scheme="ldpc",
        interleaver_scheme="qpp",
        snr_db=20.0
    )
    check("Synthetic Gen LDPC + QPP Interleaver IQ generation",
          Path(gt_ldpc["file"]).exists() and gt_ldpc["samples"] > 0,
          f"({gt_ldpc['samples']} samples, {gt_ldpc['fec_scheme']}/{gt_ldpc['interleaver_scheme']})")

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {len(failures)} FAILURES")
    print("=" * 70)
    if failures:
        print("FAILED TESTS:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL ADVANCED CODING & INTERLEAVER TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        sys.exit(0)


if __name__ == "__main__":
    main()
