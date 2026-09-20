/**
 * Mathematical utilities for calculating Attention weights and Positional Encodings
 * corresponding to Vaswani et al. (2017) "Attention Is All You Need".
 */

/**
 * Computes Positional Encoding matrix of size (seqLen x dModel).
 * Formula:
 *   PE_(pos, 2i)   = sin(pos / 10000^(2i / d_model))
 *   PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 */
export function computePositionalEncodingMatrix(seqLen = 20, dModel = 64) {
  const matrix = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = [];
    for (let i = 0; i < dModel; i += 2) {
      const divTerm = Math.pow(10000, i / dModel);
      row.push(Math.sin(pos / divTerm));
      if (i + 1 < dModel) {
        row.push(Math.cos(pos / divTerm));
      }
    }
    matrix.push(row);
  }
  return matrix;
}

/**
 * Deterministic pseudo-random number generator given a string seed.
 */
function seededRandom(seed) {
  let x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

/**
 * Tokenize input sentence cleanly.
 */
export function tokenizeText(text) {
  if (!text || !text.trim()) {
    return ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"];
  }
  return text.trim().split(/\s+/);
}

/**
 * Computes Scaled Dot-Product Attention matrix for custom tokens across multiple heads.
 * Softmax( Q * K^T / sqrt(d_k) )
 */
export function computeMultiHeadSelfAttention(text, numHeads = 8) {
  const tokens = tokenizeText(text);
  const n = tokens.length;
  const d_k = 32;

  // Multi-head attention matrices
  const heads = [];

  for (let h = 0; h < numHeads; h++) {
    // Generate pseudo Query & Key vectors for tokens tailored per head focus
    const Q = [];
    const K = [];

    for (let i = 0; i < n; i++) {
      const qVec = [];
      const kVec = [];
      const token = tokens[i].toLowerCase();
      
      for (let d = 0; d < d_k; d++) {
        let seed = (i + 1) * 31 + (d + 1) * 17 + (h + 1) * 99;
        
        // Specialize head patterns for realistic visualization:
        if (h === 0) {
          // Head 0: Local positional context (adjacent words)
          qVec.push((i / n) + (d % 2 === 0 ? 1 : -1));
          kVec.push((i / n) + (d % 2 === 0 ? 1 : -1));
        } else if (h === 1) {
          // Head 1: Coreference resolution (e.g. "it" -> "animal" / "street")
          const isIt = token.includes("it");
          const isNoun = token.includes("animal") || token.includes("street");
          qVec.push(isIt ? 2.5 : seededRandom(seed) - 0.5);
          kVec.push(isNoun ? 2.5 : seededRandom(seed + 1) - 0.5);
        } else if (h === 2) {
          // Head 2: Verb-object association ("cross" -> "street")
          const isVerb = token.includes("cross");
          const isObj = token.includes("street");
          qVec.push(isVerb ? 2.0 : seededRandom(seed) - 0.5);
          kVec.push(isObj ? 2.0 : seededRandom(seed + 2) - 0.5);
        } else {
          // General semantic heads
          let val = 0;
          for (let c = 0; c < token.length; c++) {
            val += token.charCodeAt(c) * (d + 1);
          }
          qVec.push((Math.sin(val + h) * 2));
          kVec.push((Math.cos(val * 0.5 + h * 2) * 2));
        }
      }
      Q.push(qVec);
      K.push(kVec);
    }

    // Compute raw scores = Q * K^T / sqrt(d_k)
    const rawScores = [];
    const scale = Math.sqrt(d_k);

    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let d = 0; d < d_k; d++) {
          dot += Q[i][d] * K[j][d];
        }
        row.push(dot / scale);
      }
      rawScores.push(row);
    }

    // Softmax row-wise
    const attnMatrix = [];
    for (let i = 0; i < n; i++) {
      const row = rawScores[i];
      const maxVal = Math.max(...row);
      const exps = row.map((val) => Math.exp(val - maxVal));
      const sumExps = exps.reduce((acc, curr) => acc + curr, 0);
      attnMatrix.push(exps.map((val) => val / sumExps));
    }

    heads.push({
      headIndex: h,
      matrix: attnMatrix,
    });
  }

  // Calculate average attention matrix across all heads
  const avgMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let h = 0; h < numHeads; h++) {
        sum += heads[h].matrix[i][j];
      }
      row.push(sum / numHeads);
    }
    avgMatrix.push(row);
  }

  return { tokens, heads, avgMatrix };
}

/**
 * Returns a lower triangular Causal Mask matrix.
 */
export function getCausalMaskMatrix(size = 6) {
  const matrix = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) {
      row.push(j <= i ? 1 : 0);
    }
    matrix.push(row);
  }
  return matrix;
}
