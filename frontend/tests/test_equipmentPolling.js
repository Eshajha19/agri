import { renderHook, act } from "@testing-library/react-hooks";
import useInterval from "../frontend/hooks/useInterval";

jest.useFakeTimers();

test("useInterval calls callback repeatedly", () => {
  const callback = jest.fn();
  renderHook(() => useInterval(callback, 1000));

  act(() => {
    jest.advanceTimersByTime(3000);
  });

  expect(callback).toHaveBeenCalledTimes(3);
});
