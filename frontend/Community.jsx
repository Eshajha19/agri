import React, { useState, useEffect } from "react";
import { 
  MessageSquare, 
  ThumbsUp, 
  Share2, 
  Plus, 
  Search, 
  Filter, 
  User, 
  MapPin, 
  Clock, 
  Tag, 
  MoreVertical,
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
  increment
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

const Community = () => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState("all");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCommentsModal, setShowCommentsModal] = useState(null); // stores the post object
  const [newPost, setNewPost] = useState({ content: "", category: "general" });
  const [newComment, setNewComment] = useState("");
  const [postComments, setPostComments] = useState([]);
  const [commentsLoading, setCommentsLoading] = useState(false);
  const [showP2PChat, setShowP2PChat] = useState(null); // stores the recipient object
  const [authorsData, setAuthorsData] = useState({});
  
  // Fetch author data (reputation, badges) for posts and comments
  useEffect(() => {
    const fetchAuthors = async () => {
      const authorIds = new Set([
        ...posts.map(p => p.userId),
        ...postComments.map(c => c.userId)
      ].filter(Boolean));

      const newAuthorsData = { ...authorsData };
      let changed = false;

      for (const id of authorIds) {
        if (!newAuthorsData[id]) {
          try {
            const userDoc = await getDocs(query(collection(db, "users"), where("uid", "==", id)));
            if (!userDoc.empty) {
              newAuthorsData[id] = userDoc.docs[0].data();
              changed = true;
            }
          } catch (err) {
            console.error("Error fetching author data:", err);
          }
        }
      }

      if (changed) setAuthorsData(newAuthorsData);
    };

    if (posts.length > 0 || postComments.length > 0) {
      fetchAuthors();
    }
  }, [posts, postComments]);
  
  // Mock verification check for demo
  const isVerified = (userId) => {
    // For demo: all users with 'a' in their ID or the word 'farmer' are verified
    return userId.length > 10 || userId.includes('farmer');
  };

  const currentUser = isFirebaseConfigured() ? auth?.currentUser : null;

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }
    let q = query(collection(db, "posts"), orderBy("createdAt", "desc"));
    
    if (activeCategory !== "all") {
      q = query(collection(db, "posts"), where("category", "==", activeCategory), orderBy("createdAt", "desc"));
    }

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const docs = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setPosts(docs);
      setLoading(false);
    }, (error) => {
      console.error("Error fetching posts:", error);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [activeCategory]);

  const handleCreatePost = async (e) => {
    e.preventDefault();
    if (!isFirebaseConfigured() || !currentUser || !newPost.content.trim()) return;

    try {
      await addDoc(collection(db, "posts"), {
        userId: currentUser.uid,
        userName: currentUser.displayName || currentUser.email.split('@')[0],
        userEmail: currentUser.email,
        content: newPost.content,
        category: newPost.category,
        region: "Maharashtra", // Hardcoded for demo, could be from user profile
        likes: [],
        commentsCount: 0,
        createdAt: Timestamp.now()
      });

      // Award +10 reputation for starting a discussion
      await updateDoc(doc(db, "users", currentUser.uid), {
        reputation: increment(10)
      });

      setNewPost({ content: "", category: "general" });
      setShowCreateModal(false);
      
      // Award points for creating a post
      const userRef = doc(db, "users", currentUser.uid);
      await updateDoc(userRef, { reputation: increment(10) });
    } catch (err) {
      console.error("Error creating post:", err);
    }
  };

  const handleLikePost = async (post) => {
    if (!isFirebaseConfigured() || !currentUser) return;
    const postRef = doc(db, "posts", post.id);
    const isLiked = post.likes?.includes(currentUser.uid);

    try {
      await updateDoc(postRef, {
        likes: isLiked ? arrayRemove(currentUser.uid) : arrayUnion(currentUser.uid)
      });

      // Update post author's reputation (+10 for like, -10 if unliked)
      if (post.userId !== currentUser.uid) {
        // We use the author's userId from the post
        const authorRef = doc(db, "users", post.userId);
        await updateDoc(authorRef, {
          reputation: increment(isLiked ? -10 : 10)
        });
      }
    } catch (err) {
      console.error("Error liking post:", err);
    }
  };

  const openComments = async (post) => {
    setShowCommentsModal(post);
    setCommentsLoading(true);
    try {
      const q = query(
        collection(db, "comments"), 
        where("postId", "==", post.id), 
        orderBy("createdAt", "asc")
      );
      const snapshot = await getDocs(q);
      const docs = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setPostComments(docs);
    } catch (err) {
      console.error("Error fetching comments:", err);
    } finally {
      setCommentsLoading(false);
    }
  };

  const handleAddComment = async (e) => {
    e.preventDefault();
    if (!isFirebaseConfigured() || !currentUser || !newComment.trim() || !showCommentsModal) return;

    const postId = showCommentsModal.id;
    try {
      await addDoc(collection(db, "comments"), {
        postId: postId,
        userId: currentUser.uid,
        userName: currentUser.displayName || currentUser.email.split('@')[0],
        text: newComment,
        upvotes: [],
        downvotes: [],
        createdAt: Timestamp.now()
      });

      // Award +5 reputation for posting a comment
      await updateDoc(doc(db, "users", currentUser.uid), {
        reputation: increment(5)
      });

      // Update post comment count
      const postRef = doc(db, "posts", postId);
      await updateDoc(postRef, {
        commentsCount: increment(1)
      });

      setNewComment("");
      // Refresh comments locally for now or use another listener
      openComments(showCommentsModal);

      // Award points for commenting
      const userRef = doc(db, "users", currentUser.uid);
      await updateDoc(userRef, { reputation: increment(5) });
    } catch (err) {
      console.error("Error adding comment:", err);
    }
  };

  const handleVoteComment = async (comment, voteType) => {
    if (!isFirebaseConfigured() || !currentUser) return;
    
    const commentRef = doc(db, "comments", comment.id);
    const hasUpvoted = comment.upvotes?.includes(currentUser.uid);
    const hasDownvoted = comment.downvotes?.includes(currentUser.uid);
    
    let reputationChange = 0;
    let updates = {};

    if (voteType === 'up') {
      if (hasUpvoted) {
        updates.upvotes = arrayRemove(currentUser.uid);
        reputationChange = -10;
      } else {
        updates.upvotes = arrayUnion(currentUser.uid);
        reputationChange = 10;
        if (hasDownvoted) {
          updates.downvotes = arrayRemove(currentUser.uid);
          reputationChange += 2; // recover the -2 from downvote
        }
      }
    } else {
      if (hasDownvoted) {
        updates.downvotes = arrayRemove(currentUser.uid);
        reputationChange = 2;
      } else {
        updates.downvotes = arrayUnion(currentUser.uid);
        reputationChange = -2;
        if (hasUpvoted) {
          updates.upvotes = arrayRemove(currentUser.uid);
          reputationChange -= 10; // remove the +10 from upvote
        }
      }
    }

    try {
      await updateDoc(commentRef, updates);
      
      // Update comment author's reputation
      if (comment.userId !== currentUser.uid) {
        await updateDoc(doc(db, "users", comment.userId), {
          reputation: increment(reputationChange)
        });
      }
      
      // Refresh comments
      openComments(showCommentsModal);
    } catch (err) {
      console.error("Error voting on comment:", err);
    }
  };

  const getBadgeIcon = (reputation) => {
    if (reputation >= 500) return "🥇";
    if (reputation >= 200) return "🥈";
    if (reputation >= 50) return "🥉";
    return null;
  };

  const getBadgeTitle = (reputation) => {
    if (reputation >= 500) return "Master Agriculturist";
    if (reputation >= 200) return "Farming Expert";
    if (reputation >= 50) return "Active Contributor";
    return "";
  };

  const filteredPosts = posts.filter(post => 
    post.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    post.userName.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="community-container">
       <header className="community-header">
         <div className="header-top">
           <h1><MessageSquare className="title-icon" /> <span className="notranslate">Farmer Community</span></h1>
           <p>Share knowledge, ask questions, and grow together with farmers across India</p>
         </div>
        
        <div className="header-controls">
          <div className="search-bar">
            <Search size={20} className="search-icon" />
            <input 
              type="text" 
              placeholder="Search discussions, topics, or farmers..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <button className="create-post-btn" onClick={() => setShowCreateModal(true)} disabled={!isFirebaseConfigured()}>
            <Plus size={20} /> Create Discussion
          </button>
        </div>

        <div className="category-tabs">
          {CATEGORIES.map(cat => (
            <button 
              key={cat.id} 
              className={`cat-tab ${activeCategory === cat.id ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat.id)}
            >
              {activeCategory === cat.id && <Tag size={14} />}
              {cat.label}
            </button>
          ))}
        </div>
      </header>

      <main className="community-feed">
        {loading ? (
          <Loader message="Loading discussions..." />
         ) : filteredPosts.length === 0 ? (
          <div className="empty-feed">
            <MessageSquare size={64} className="empty-icon" />
            <h3>No discussions found</h3>
            {!isFirebaseConfigured() ? (
              <p>Community features require Firebase configuration. Please check your Firebase settings.</p>
            ) : (
              <p>Be the first one to start a conversation in this category!</p>
            )}
            {isFirebaseConfigured() && <button className="btn-secondary" onClick={() => setShowCreateModal(true)}>Start a Discussion</button>}
          </div>
        ) : (
          <div className="posts-grid">
            {filteredPosts.map(post => (
              <div key={post.id} className="post-card">
                <div className="post-header">
                  <div className="user-info">
                    <div className="user-avatar">
                      {post.userName ? post.userName[0].toUpperCase() : "U"}
                      {isVerified(post.userId) && <ShieldCheck className="verified-badge-community" size={14} />}
                    </div>
                    <div>
                      <div className="user-name-wrapper">
                        <h3>{post.userName}</h3>
                        {authorsData[post.userId]?.reputation >= 50 && (
                          <span className="expert-badge" title={getBadgeTitle(authorsData[post.userId]?.reputation)}>
                            {getBadgeIcon(authorsData[post.userId]?.reputation)}
                          </span>
                        )}
                      </div>
                      <div className="post-meta">
                        <span><MapPin size={12} /> {post.region || "All India"}</span>
                        <span><Clock size={12} /> {post.createdAt?.toDate ? post.createdAt.toDate().toLocaleDateString() : "Recent"}</span>
                      </div>
                    </div>
                  </div>
                  <div className="post-category" style={{ backgroundColor: CATEGORIES.find(c => c.id === post.category)?.color + '20', color: CATEGORIES.find(c => c.id === post.category)?.color }}>
                    {CATEGORIES.find(c => c.id === post.category)?.label}
                  </div>
                </div>
                
                <div className="post-content">
                  <p>{post.content}</p>
                </div>

                 <div className="post-actions">
                  <button 
                    className={`action-btn ${post.likes?.includes(currentUser?.uid) ? 'liked' : ''}`}
                    onClick={() => handleLikePost(post)}
                    disabled={!isFirebaseConfigured() || !currentUser}
                  >
                    <ThumbsUp size={18} fill={post.likes?.includes(currentUser?.uid) ? "currentColor" : "none"} />
                    {post.likes?.length || 0}
                  </button>
                  <button className="action-btn" onClick={() => openComments(post)}>
                    <MessageSquare size={18} />
                    {post.commentsCount || 0}
                  </button>
                  {currentUser && (
                    <button 
                      className="action-btn p2p-action-btn" 
                      onClick={() => setShowP2PChat({ userId: post.userId, userName: post.userName })}
                      title="Send Private Encrypted Message"
                    >
                      <MessageCircle size={18} />
                      <span className="p2p-label">Chat</span>
                    </button>
                  )}
                  <button className="action-btn">
                    <Share2 size={18} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Create Post Modal */}
      {showCreateModal && (
        <div className="modal-overlay">
          <div className="modal-card post-modal">
            <div className="modal-header">
              <h3>Start a New Discussion</h3>
              <button className="close-btn" onClick={() => setShowCreateModal(false)}><X /></button>
            </div>
            <form onSubmit={handleCreatePost}>
              <div className="form-group">
                <label>Category</label>
                <select 
                  value={newPost.category}
                  onChange={(e) => setNewPost({...newPost, category: e.target.value})}
                >
                  {CATEGORIES.filter(c => c.id !== 'all').map(cat => (
                    <option key={cat.id} value={cat.id}>{cat.label}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Your Message</label>
                <textarea 
                  rows="5" 
                  placeholder={isFirebaseConfigured() ? "What's on your mind? Ask a question or share an experience..." : "Firebase not configured - cannot create posts"}
                  value={newPost.content}
                  onChange={(e) => setNewPost({...newPost, content: e.target.value})}
                  required
                  disabled={!isFirebaseConfigured()}
                ></textarea>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn-cancel" onClick={() => setShowCreateModal(false)}><span className="notranslate">Cancel</span></button>
                <button type="submit" className="btn-submit"><span className="notranslate">Post to Community</span></button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Comments Modal */}
      {showCommentsModal && (
        <div className="modal-overlay">
          <div className="modal-card comments-modal">
            <div className="modal-header">
              <h3>Comments</h3>
              <button className="close-btn" onClick={() => setShowCommentsModal(null)}><X /></button>
            </div>
            
            <div className="original-post-context">
              <div className="user-info mini">
                <div className="user-avatar mini">{showCommentsModal.userName ? showCommentsModal.userName[0].toUpperCase() : "U"}</div>
                <span>{showCommentsModal.userName}</span>
              </div>
              <p>{showCommentsModal.content}</p>
            </div>

            <div className="comments-list">
              {commentsLoading ? (
                <div className="mini-loader-wrap"><Loader message="" /></div>
              ) : postComments.length === 0 ? (
                <p className="no-comments">No comments yet. Be the first to reply!</p>
              ) : (
                postComments.map(comment => (
                  <div key={comment.id} className="comment-item">
                    <div className="comment-header">
                      <div className="comment-user-info">
                        <strong>{comment.userName}</strong>
                        {authorsData[comment.userId]?.reputation >= 50 && (
                          <span className="expert-badge-mini" title={getBadgeTitle(authorsData[comment.userId]?.reputation)}>
                            {getBadgeIcon(authorsData[comment.userId]?.reputation)}
                          </span>
                        )}
                        <span className="reputation-text">{authorsData[comment.userId]?.reputation || 0} pts</span>
                      </div>
                      <span>{comment.createdAt?.toDate ? comment.createdAt.toDate().toLocaleDateString() : "Recent"}</span>
                    </div>
                    <p>{comment.text}</p>
                    <div className="comment-votes">
                      <button 
                        className={`vote-btn up ${comment.upvotes?.includes(currentUser?.uid) ? 'active' : ''}`}
                        onClick={() => handleVoteComment(comment, 'up')}
                      >
                        <ThumbsUp size={14} fill={comment.upvotes?.includes(currentUser?.uid) ? "currentColor" : "none"} />
                        {comment.upvotes?.length || 0}
                      </button>
                      <button 
                        className={`vote-btn down ${comment.downvotes?.includes(currentUser?.uid) ? 'active' : ''}`}
                        onClick={() => handleVoteComment(comment, 'down')}
                      >
                        <ThumbsUp size={14} style={{ transform: 'rotate(180deg)' }} fill={comment.downvotes?.includes(currentUser?.uid) ? "currentColor" : "none"} />
                        {comment.downvotes?.length || 0}
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

             <form className="comment-form" onSubmit={handleAddComment}>
               <input 
                 type="text" 
                 placeholder={isFirebaseConfigured() && currentUser ? "Write a reply..." : "Login to comment"}
                 value={newComment}
                 onChange={(e) => setNewComment(e.target.value)}
                 required
                 disabled={!isFirebaseConfigured() || !currentUser}
               />
               <button type="submit" className="send-btn" disabled={!isFirebaseConfigured() || !currentUser}><Send size={18} /></button>
             </form>
          </div>
        </div>
      )}
      {/* P2P Chat Modal */}
      {showP2PChat && (
        <div className="modal-overlay chat-modal-overlay">
          <div className="modal-card p2p-chat-modal-wrapper" onClick={(e) => e.stopPropagation()}>
            <P2PChat 
              recipient={showP2PChat} 
              onClose={() => setShowP2PChat(null)} 
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Community;
