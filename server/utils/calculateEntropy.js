/**
 * Calculate Shannon entropy for a given response count distribution.
 * 
 * @param {Object<string, number>} responseCounts - Mapping of answer => count
 * @returns {number} entropy value in bits
 */
function calculateEntropy(responseCounts) {
  if (
    !responseCounts ||
    typeof responseCounts !== 'object' ||
    Array.isArray(responseCounts)
  ) {
    return 0;
  }

  const counts = Object.values(responseCounts).filter(
    (c) => typeof c === 'number' && c > 0
  );

  const total = counts.reduce((sum, count) => sum + count, 0);
  if (total === 0) return 0;

  let entropy = 0;
  for (const count of counts) {
    const p = count / total;
    entropy -= p * Math.log2(p);
  }

  return entropy;
}

module.exports = calculateEntropy;
