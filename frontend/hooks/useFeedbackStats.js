import { useState, useEffect } from "react";
import {
  collection,
  query,
  getDocs,
  onSnapshot,
  Timestamp,
} from "firebase/firestore";
import { db, isFirebaseConfigured } from "../lib/firebase";

export default function useFeedbackStats() {
  const [avgRating, setAvgRating] = useState(0);
  const [totalFeedbacks, setTotalFeedbacks] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }

    const feedbackRef = collection(db, "feedback");
    const q = query(feedbackRef);

    // Subscribe to real-time updates
    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        let total = 0;
        let ratingSum = 0;

        snapshot.forEach((doc) => {
          const data = doc.data();
          if (data.rating) {
            ratingSum += data.rating;
            total++;
          }
        });

        setTotalFeedbacks(total);
        setAvgRating(total > 0 ? ratingSum / total : 0);
        setLoading(false);
      },
      (error) => {
        console.error("Error fetching feedback stats:", error);
        setLoading(false);
      }
    );

    // Fallback: also try one-time fetch if snapshot fails
    const fallbackFetch = async () => {
      try {
        const snapshot = await getDocs(q);
        let total = 0;
        let ratingSum = 0;

        snapshot.forEach((doc) => {
          const data = doc.data();
          if (data.rating) {
            ratingSum += data.rating;
            total++;
          }
        });

        setTotalFeedbacks(total);
        setAvgRating(total > 0 ? ratingSum / total : 0);
      } catch (err) {
        console.error("Fallback fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    // Start with fallback fetch
    fallbackFetch();

    return () => unsubscribe();
  }, []);

  return { avgRating, totalFeedbacks, loading };
}
