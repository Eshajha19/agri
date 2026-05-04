import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  MessageSquare,
  ThumbsUp,
  Share2,
  Plus,
  Search,
  MapPin,
  Clock,
  Send,
  X,
  ShieldCheck,
  MessageCircle
} from "lucide-react";

import P2PChat from "./P2PChat";
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
  getDocs,
  increment,
  limit,
  startAfter
} from "firebase/firestore";

import Loader from "./Loader";
import "./Community.css";

/* ================= CONFIG ================= */

const CATEGORIES = [
  { id: "all", label: "All Topics", color: "#64748b" },
  { id: "general", label: "General Discussion", color: "#3b82f6" },
  { id: "crops", label: "Crop Management", color: "#10b981" },
  { id: "pests", label: "Pest Control", color: "#ef4444" },
  { id: "market", label: "Market Prices", color: "#f59e0b" },
  { id: "success", label: "Success Stories", color: "#8b5cf6" },
];

/* ================= HELPERS ================= */

const avatarColor = (name = "U") => {
  const colors = ["#f87171", "#60a5fa", "#34d399", "#fbbf24"];
  return colors[name.charCodeAt(0) % colors.length];
};

const timeAgo = (date) => {
  if (!date) return "Just now";
  const diff = Math.floor((Date.now() - date) / 1000);

  const units = {
    year: 31536000,
    month: 2592000,
    day: 86400,
    hour: 3600,
    minute: 60
  };

  for (let u in units) {
    const v = Math.floor(diff / units[u]);
    if (v > 0) return `${v} ${u}${v > 1 ? "s" : ""} ago`;
  }
  return "Just now";
};

/* ================= MAIN COMPONENT ================= */

const Community = () => {

  const [posts, setPosts] = useState([]);
  const [lastDoc, setLastDoc] = useState(null);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");

  const [showCreate, setShowCreate] = useState(false);
  const [activePost, setActivePost] = useState(null);
  const [comments, setComments] = useState([]);
  const [commentText, setCommentText] = useState("");

  const [chatUser, setChatUser] = useState(null);

  const unsubComments = useRef(null);

  const currentUser = auth?.currentUser;

  /* ================= POSTS REALTIME ================= */

  useEffect(() => {
    if (!isFirebaseConfigured()) return;

    setLoading(true);

    let qRef = query(
      collection(db, "posts"),
      orderBy("createdAt", "desc"),
      limit(10)
    );

    if (category !== "all") {
      qRef = query(
        collection(db, "posts"),
        where("category", "==", category),
        orderBy("createdAt", "desc"),
        limit(10)
      );
    }

    const unsub = onSnapshot(qRef, (snap) => {
      const data = snap.docs.map(d => ({ id: d.id, ...d.data() }));
      setPosts(data);
      setLastDoc(snap.docs[snap.docs.length - 1]);
      setLoading(false);
    });

    return () => unsub();
  }, [category]);

  /* ================= LOAD MORE ================= */

  const loadMore = async () => {
    if (!lastDoc) return;

    const qRef = query(
      collection(db, "posts"),
      orderBy("createdAt", "desc"),
      startAfter(lastDoc),
      limit(5)
    );

    const snap = await getDocs(qRef);

    const more = snap.docs.map(d => ({ id: d.id, ...d.data() }));

    setPosts(prev => [...prev, ...more]);
    setLastDoc(snap.docs[snap.docs.length - 1]);
  };

  /* ================= CREATE POST ================= */

  const handleCreatePost = async (e) => {
    e.preventDefault();
    if (!currentUser) return;

    const content = e.target.content.value;

    await addDoc(collection(db, "posts"), {
      userId: currentUser.uid,
      userName: currentUser.displayName || "Farmer",
      content,
      category,
      likes: [],
      commentsCount: 0,
      createdAt: Timestamp.now()
    });

    setShowCreate(false);
  };

  /* ================= LIKE POST ================= */

  const handleLikePost = async (post) => {
    if (!currentUser) return;

    const ref = doc(db, "posts", post.id);
    const liked = post.likes?.includes(currentUser.uid);

    await updateDoc(ref, {
      likes: liked
        ? arrayRemove(currentUser.uid)
        : arrayUnion(currentUser.uid)
    });
  };

  /* ================= COMMENTS ================= */

  const openComments = (post) => {
    setActivePost(post);

    if (unsubComments.current) unsubComments.current();

    const qRef = query(
      collection(db, "comments"),
      where("postId", "==", post.id),
      orderBy("createdAt", "asc")
    );

    unsubComments.current = onSnapshot(qRef, (snap) => {
      setComments(snap.docs.map(d => ({ id: d.id, ...d.data() })));
    });
  };

  /* ================= ADD COMMENT ================= */

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!currentUser || !commentText.trim()) return;

    await addDoc(collection(db, "comments"), {
      postId: activePost.id,
      userId: currentUser.uid,
      userName: currentUser.displayName || "Farmer",
      text: commentText,
      upvotes: [],
      downvotes: [],
      createdAt: Timestamp.now()
    });

    await updateDoc(doc(db, "posts", activePost.id), {
      commentsCount: increment(1)
    });

    setCommentText("");
  };

  /* ================= FILTER POSTS ================= */

  const filteredPosts = useMemo(() => {
    return posts.filter(p =>
      p.content.toLowerCase().includes(search.toLowerCase())
    );
  }, [posts, search]);

  /* ================= UI ================= */

  return (
    <div className="community-container">

      {/* HEADER */}
      <header className="community-header">
        <h2>🌾 Farmer Community</h2>

        <div className="search-bar">
          <Search />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search discussions..."
          />
        </div>

        <button onClick={() => setShowCreate(true)}>
          <Plus /> Post
        </button>
      </header>

      {/* POSTS */}
      <main>
        {loading ? (
          <Loader message="Loading community..." />
        ) : filteredPosts.map(post => (
          <div key={post.id} className="post-card">

            <div className="post-header">
              <div
                className="avatar"
                style={{ background: avatarColor(post.userName) }}
              >
                {post.userName?.[0]}
              </div>

              <div>
                <h4>{post.userName}</h4>
                <span>{timeAgo(post.createdAt?.toDate())}</span>
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

              <button onClick={() => setChatUser(post.userId)}>
                <MessageCircle />
              </button>

              <button>
                <Share2 />
              </button>
            </div>
          </div>
        ))}

        {posts.length > 5 && (
          <button onClick={loadMore} className="load-more">
            Load More
          </button>
        )}
      </main>

      {/* CREATE POST */}
      {showCreate && (
        <div className="modal">
          <form onSubmit={handleCreatePost}>
            <textarea name="content" placeholder="Share something..." />
            <button type="submit">Post</button>
          </form>
        </div>
      )}

      {/* COMMENTS */}
      {activePost && (
        <div className="modal">
          <div className="modal-box">

            <h3>Comments</h3>

            {comments.map(c => (
              <p key={c.id}>
                <b>{c.userName}</b>: {c.text}
              </p>
            ))}

            <form onSubmit={handleAddComment}>
              <input
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
                placeholder="Write a comment..."
              />
              <button type="submit"><Send /></button>
            </form>

            <button onClick={() => setActivePost(null)}>
              <X />
            </button>

          </div>
        </div>
      )}

      {/* CHAT */}
      {chatUser && (
        <P2PChat
          recipient={{ userId: chatUser }}
          onClose={() => setChatUser(null)}
        />
      )}

    </div>
  );
};

export default Community;
