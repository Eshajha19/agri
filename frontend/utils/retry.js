
export async function retryWithBackoff(fn, {
  retries = 3,
  baseDelay = 500, // ms
  factor = 2,
  context = "unknown"
} = {}) {
  let attempt = 0;
  while (attempt < retries) {
    try {
      return await fn();
    } catch (error) {
      if (error?.response?.status !== 429) throw error;

      const delay = baseDelay * Math.pow(factor, attempt);
      console.warn(`[retry] ${context} attempt ${attempt + 1} failed, retrying in ${delay}ms`);

      await new Promise(res => setTimeout(res, delay));
      attempt++;
    }
  }
  throw new Error(`Failed after ${retries} retries`);
}
