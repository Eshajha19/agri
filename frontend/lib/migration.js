import { db, doc, getDoc, setDoc, updateDoc } from "./firebase";
import { collection, getDocs, query, where, writeBatch } from "firebase/firestore";

/**
 * Migrates data from an anonymous user to a permanent account.
 * This is used when a guest user logs into an existing account instead of creating a new one,
 * or when account linking is not possible.
 */
export const migrateUserData = async (anonymousUid, permanentUid) => {
  if (!anonymousUid || !permanentUid || anonymousUid === permanentUid) return;

  console.log(`Starting data migration from ${anonymousUid} to ${permanentUid}`);
  const batch = writeBatch(db);

  try {
    // 1. Migrate User Profile
    const anonUserRef = doc(db, "users", anonymousUid);
    const permUserRef = doc(db, "users", permanentUid);
    
    const anonUserSnap = await getDoc(anonUserRef);
    if (anonUserSnap.exists()) {
      const anonData = anonUserSnap.data();
      // Only merge if the permanent user doesn't have these fields or if they are "default"
      const permUserSnap = await getDoc(permUserRef);
      const permData = permUserSnap.exists() ? permUserSnap.data() : {};

      const mergedData = {
        ...anonData,
        ...permData, // Permanent data wins
        uid: permanentUid, // Ensure UID is correct
        updatedAt: new Date().toISOString()
      };
      
      batch.set(permUserRef, mergedData, { merge: true });
    }

    // 2. Migrate Farming Activities
    const activitiesQuery = query(collection(db, "activities"), where("userId", "==", anonymousUid));
    const activitiesSnap = await getDocs(activitiesQuery);
    activitiesSnap.forEach((d) => {
      batch.update(doc(db, "activities", d.id), { userId: permanentUid });
    });

    // 3. Migrate Feedback
    const feedbackQuery = query(collection(db, "feedback"), where("userId", "==", anonymousUid));
    const feedbackSnap = await getDocs(feedbackQuery);
    feedbackSnap.forEach((d) => {
      batch.update(doc(db, "feedback", d.id), { userId: permanentUid });
    });

    // 4. Migrate Community Posts
    const postsQuery = query(collection(db, "posts"), where("userId", "==", anonymousUid));
    const postsSnap = await getDocs(postsQuery);
    postsSnap.forEach((d) => {
      batch.update(doc(db, "posts", d.id), { userId: permanentUid });
    });

    // 5. Migrate Comments
    const commentsQuery = query(collection(db, "comments"), where("userId", "==", anonymousUid));
    const commentsSnap = await getDocs(commentsQuery);
    commentsSnap.forEach((d) => {
      batch.update(doc(db, "comments", d.id), { userId: permanentUid });
    });

    // 6. Migrate Direct Messages
    const dmQuery = query(collection(db, "direct_messages"), where("senderId", "==", anonymousUid));
    const dmSnap = await getDocs(dmQuery);
    dmSnap.forEach((d) => {
      batch.update(doc(db, "direct_messages", d.id), { senderId: permanentUid });
    });

    // Commit all changes
    await batch.commit();
    console.log("Migration successful!");
    return { success: true };
  } catch (error) {
    console.error("Migration failed:", error);
    throw error;
  }
};
