import { describe, it, expect, vi } from "vitest";
import { synchronizeTranslation } from "../lib/googleTranslate";

describe("Google Translate widget synchronization", () => {
  it("retries via the robust fallback when the fast path fails", async () => {
    let retryCount = 0;
    const mockApply = vi.fn(() => false);
    const mockRobust = vi.fn((lang, opts) => {
      retryCount++;
      opts.onError();
    });

    await synchronizeTranslation("hi", mockApply, mockRobust);

    expect(mockRobust).toHaveBeenCalled();
    expect(retryCount).toBeGreaterThan(0);
  });

  it("does not call the robust fallback when the fast path succeeds", async () => {
    const mockApply = vi.fn(() => true);
    const mockRobust = vi.fn();

    await synchronizeTranslation("en", mockApply, mockRobust);

    expect(mockApply).toHaveBeenCalledWith("en");
    expect(mockRobust).not.toHaveBeenCalled();
  });

  it("skips when no language code is provided", async () => {
    const mockApply = vi.fn(() => true);
    const mockRobust = vi.fn();

    await synchronizeTranslation("", mockApply, mockRobust);

    expect(mockApply).not.toHaveBeenCalled();
    expect(mockRobust).not.toHaveBeenCalled();
  });
});
