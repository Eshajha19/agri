test("falls back to Firebase when Google fails", async () => {
  const service = new TranslationService();
  service.googleTranslateProvider = jest.fn(() => { throw new Error("fail"); });
  service.firebaseTranslateProvider = jest.fn(() => "Hola");
  const result = await service.translate("Hello", "es");
  expect(result).toBe("Hola");
});

test("falls back to local dictionary when all providers fail", async () => {
  const service = new TranslationService();
  service.googleTranslateProvider = jest.fn(() => { throw new Error("fail"); });
  service.firebaseTranslateProvider = jest.fn(() => { throw new Error("fail"); });
  const result = await service.translate("Hello", "hi");
  expect(result).toBe("नमस्ते");
});
