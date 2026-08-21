import { describe, expect, it } from "vitest";
import { LANGUAGE_OPTIONS, SUPPORTED_LANGUAGE_CODES } from "./languageOptions";

describe("supported languages", () => {
  it("exposes every requested language exactly once", () => {
    expect(LANGUAGE_OPTIONS.map(({ value }) => value)).toEqual(SUPPORTED_LANGUAGE_CODES);
    expect(SUPPORTED_LANGUAGE_CODES).toEqual([
      "en", "hi", "mr", "bn", "ta", "te", "gu", "pa", "kn", "ml", "or", "as"
    ]);
  });
});