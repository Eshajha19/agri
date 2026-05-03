import React, { useState, useEffect, useMemo } from "react";
import {
  MessageSquare,
  ThumbsUp,
  Share2,
  Plus,
  Search,
  MapPin,
  Clock,
  Tag,
  Send,
  X
} from "lucide-react";

import { auth, db, isFirebaseConfigured } from "./lib/firebase";
import {
  collection,
  addDoc,
  query,
  orderBy,
  onSnapshot,
  doc,
  updateDoc,
  arrayUnion,
  arrayRemove,
  where,
  Timestamp,
  increment,
  limit,
  startAfter
} from "firebase/firestore";

import Loader from "./Loader";
import "./Community.css";

const CATEGORIES = [
  { id: "all", label: "All Topics", color: "#64748b" },
  { id: "general", label: "General Discussion", color: "#3b82f6" },
  { id: "crops", label: "Crop Management", color: "#10b981" },
  { id: "pests", label: "Pest Control", color: "#ef4444" },
  { id: "market", label: "Market Prices", color: "#f59e0b" },
  { id: "success", label: "Success Stories", color: "#8b5cf6" },
];

const getAvatarColor = (name = "U") => {
  const colors = ["#f87171", "#60a5fa", "#34d399", "#fbbf24"];
  return colors[name.charCodeAt(0) % colors.length];
};

const timeAgo = (date) => {
  if (!date) return "Recent";
  const seconds = Math.floor((new Date() - date) / 1000);

  const intervals = {
    year: 31536000,
    month: 2592000,
    day: 86400,
    hour: 3600,
    minute: 60,
  };

  for (let key in intervals) {
    const value = Math.floor(seconds / intervals[key]);
    if (value >= 1) return `${value} ${key}${value > 1 ? "s" : ""} ago`;
  }
  return "Just now";
};

const Community = () => {
  const [posts, setPosts] = useState([]);
  const [lastDoc, setLastDoc] = useState(null);
  const [loading, setLoading] = useState(true);

  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCommentsModal, setShowCommentsModal] = useState(null);

  const [newPost, setNewPost] = useState({ content: "", category: "general" });
  const [newComment, setNewComment] = useState("");

  const [postComments, setPostComments] = useState([]);
  const [commentsLoading, setCommentsLoading] = useState(false);

  const currentUser = isFirebaseConfigured() ? auth?.currentUser : null;

  // 🔥 FETCH POSTS (REAL-TIME + PAGINATION)
  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }

    let q = query(
      collection(db, "posts"),
      orderBy("createdAt", "desc"),
      limit(10)
    );

    if (activeCategory !== "all") {
      q = query(
        collection(db, "posts"),
        where("category", "==", activeCategory),
        orderBy("createdAt", "desc"),
        limit(10)
      );
    }

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      setLastDoc(snapshot.docs[snapshot.docs.length - 1]);
      setPosts(docs);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [activeCategory]);

  // 🔥 LOAD MORE
  const loadMore = async () => {
    if (!lastDoc) return;

    let q = query(
      collection(db, "posts"),
      orderBy("createdAt", "desc"),
      startAfter(lastDoc),
      limit(5)
    );

    const snapshot = await onSnapshot(q, () => {});
    const docs = snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));

    setLastDoc(snapshot.docs[snapshot.docs.length - 1]);
    setPosts((prev) => [...prev, ...docs]);
  };

  // 🔥 CREATE POST
  const handleCreatePost = async (e) => {
    e.preventDefault();
    if (!currentUser || !newPost.content.trim()) return;

    await addDoc(collection(db, "posts"), {
      userId: currentUser.uid,
      userName:
        currentUser.displayName || currentUser.email.split("@")[0],
      content: newPost.content,
      category: newPost.category,
      region: "India",
      likes: [],
      commentsCount: 0,
      createdAt: Timestamp.now(),
    });

    setNewPost({ content: "", category: "general" });
    setShowCreateModal(false);
  };

  // 🔥 LIKE POST
  const handleLikePost = async (post) => {
    if (!currentUser) return;

    const ref = doc(db, "posts", post.id);
    const liked = post.likes?.includes(currentUser.uid);

    await updateDoc(ref, {
      likes: liked
        ? arrayRemove(currentUser.uid)
        : arrayUnion(currentUser.uid),
    });
  };

  // 🔥 REAL-TIME COMMENTS
  const openComments = (post) => {
    setShowCommentsModal(post);
    setCommentsLoading(true);

    const q = query(
      collection(db, "comments"),
      where("postId", "==", post.id),
      orderBy("createdAt", "asc")
    );

    return onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));
      setPostComments(docs);
      setCommentsLoading(false);
    });
  };

  // 🔥 ADD COMMENT
  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!currentUser || !newComment.trim()) return;

    await addDoc(collection(db, "comments"), {
      postId: showCommentsModal.id,
      userId: currentUser.uid,
      userName:
        currentUser.displayName || currentUser.email.split("@")[0],
      text: newComment,
      createdAt: Timestamp.now(),
    });

    await updateDoc(doc(db, "posts", showCommentsModal.id), {
      commentsCount: increment(1),
    });

    setNewComment("");
  };

  // 🔥 SEARCH OPTIMIZATION
  const filteredPosts = useMemo(() => {
    return posts.filter((p) =>
      p.content.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [posts, searchQuery]);

  return (
    <div className="community-container">
      <header className="community-header">
        <h1><MessageSquare /> Farmer Community</h1>

        <div className="search-bar">
          <Search />
          <input
            placeholder="Search discussions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <button onClick={() => setShowCreateModal(true)}>
          <Plus /> Create
        </button>
      </header>

      <main>
        {loading ? (
          <Loader message="Loading..." />
        ) : filteredPosts.map((post) => (
          <div key={post.id} className="post-card">
            <div className="user-info">
              <div
                className="avatar"
                style={{ background: getAvatarColor(post.userName) }}
              >
                {post.userName[0]}
              </div>

              <div>
                <h3>{post.userName}</h3>
                <span>
                  <Clock />{" "}
                  {timeAgo(post.createdAt?.toDate())}
                </span>
              </div>
            </div>

            <p>{post.content}</p>

            <div className="actions">
              <button onClick={() => handleLikePost(post)}>
                <ThumbsUp /> {post.likes?.length || 0}
              </button>

              <button onClick={() => openComments(post)}>
                <MessageSquare /> {post.commentsCount || 0}
              </button>
            </div>
          </div>
        ))}

        <button onClick={loadMore}>Load More</button>
      </main>

      {/* CREATE MODAL */}
      {showCreateModal && (
        <div className="modal">
          <form onSubmit={handleCreatePost}>
            <textarea
              value={newPost.content}
              onChange={(e) =>
                setNewPost({ ...newPost, content: e.target.value })
              }
            />
            <button type="submit">Post</button>
          </form>
        </div>
      )}

      {/* COMMENTS MODAL */}
      {showCommentsModal && (
        <div className="modal">
          <h3>Comments</h3>

          {commentsLoading ? (
            <Loader />
          ) : postComments.map((c) => (
            <p key={c.id}>
              <strong>{c.userName}</strong>: {c.text}
            </p>
          ))}

          <form onSubmit={handleAddComment}>
            <input
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
            />
            <button type="submit"><Send /></button>
          </form>
        </div>
      )}
    </div>
  );
};

export default Community;