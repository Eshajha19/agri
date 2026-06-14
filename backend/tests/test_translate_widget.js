test("Google Translate widget retries on failure", async () => {
  let retryCount = 0;
  const mockApply = jest.fn(() => false);
  const mockRobust = jest.fn((lang, opts) => {
    retryCount++;
    opts.onError();
  });

  await synchronizeTranslation("hi", mockApply, mockRobust);

  expect(mockRobust).toHaveBeenCalled();
  expect(retryCount).toBeGreaterThan(0);
});
