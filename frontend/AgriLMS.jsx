import React, { useState, useEffect, useCallback } from 'react';
import './AgriLMS.css';
import { Play, CheckCircle, Award, BookOpen, Clock, Download, ChevronRight, MessageCircle, Loader } from 'lucide-react';
import jsPDF from 'jspdf';
import SoilChatbot from './SoilChatbot';
import apiClient from './services/api';

// ---------------------------------------------------------------------------
// Course catalogue — mirrors backend COURSES dict in backend/routers/lms.py.
// Lesson IDs must match exactly; the backend validates them server-side.
// ---------------------------------------------------------------------------
const COURSES = [
  {
    id: 'soil-health',
    title: 'Advanced Soil Management',
    category: 'Soil',
    duration: '45 mins',
    lessons: [
      { id: 's1', title: 'Testing Soil pH at Home',     duration: '10:00', videoUrl: 'https://www.youtube.com/embed/5_gYbLGiVMI' },
      { id: 's2', title: 'Organic Matter Enrichment',   duration: '15:00', videoUrl: 'https://www.youtube.com/embed/elEuxFzbTO0' },
      { id: 's3', title: 'Crop Rotation Basics',        duration: '20:00', videoUrl: 'https://www.youtube.com/embed/3QLYFg4NIN8' },
    ],
  },
  {
    id: 'pest-control',
    title: 'Organic Pest Management',
    category: 'Pest Control',
    duration: '30 mins',
    lessons: [
      { id: 'p1', title: 'Natural Insecticides',        duration: '12:00', videoUrl: 'https://www.youtube.com/embed/ZyvcmpyD7FM' },
      { id: 'p2', title: 'Biological Control Agents',   duration: '18:00', videoUrl: 'https://www.youtube.com/embed/g6LMw9I6rxU' },
    ],
  },
  {
    id: 'modern-tools',
    title: 'Drones in Agriculture',
    category: 'Technology',
    duration: '25 mins',
    lessons: [
      { id: 't1', title: 'Drone Mapping Basics',        duration: '10:00', videoUrl: 'https://www.youtube.com/embed/QtXhHZP5SSY' },
      { id: 't2', title: 'Precision Spraying',          duration: '15:00', videoUrl: 'https://www.youtube.com/embed/-0rAAqVeCG8' },
    ],
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute course completion percentage from server-authoritative progress.
 * `serverProgress` shape: { [courseId]: { lessons: { [lessonId]: true } } }
 */
function getCourseProgress(course, serverProgress) {
  const done = serverProgress[course.id]?.lessons ?? {};
  const completed = course.lessons.filter(l => done[l.id] === true).length;
  return Math.round((completed / course.lessons.length) * 100);
}

/**
 * Generate and download a PDF certificate using data returned by the backend.
 * The recipient name comes from the server — never from localStorage or a
 * hardcoded string.
 */
function downloadCertificate({ recipient_name, course_title, completed_at, cert_id }) {
  const doc = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' });

  doc.setFillColor(245, 247, 241);
  doc.rect(0, 0, 297, 210, 'F');

  doc.setDrawColor(46, 125, 50);
  doc.setLineWidth(5);
  doc.rect(10, 10, 277, 190);

  doc.setTextColor(46, 125, 50);
  doc.setFontSize(40);
  doc.text('Certificate of Completion', 148.5, 50, { align: 'center' });

  doc.setTextColor(33, 33, 33);
  doc.setFontSize(20);
  doc.text('This is to certify that', 148.5, 80, { align: 'center' });

  doc.setFontSize(30);
  doc.setFont('helvetica', 'bold');
  // Recipient name is sourced from the server-verified Firestore user profile.
  doc.text(recipient_name, 148.5, 105, { align: 'center' });

  doc.setFontSize(20);
  doc.setFont('helvetica', 'normal');
  doc.text('has successfully completed the course', 148.5, 130, { align: 'center' });

  doc.setFontSize(25);
  doc.setTextColor(46, 125, 50);
  doc.text(course_title, 148.5, 155, { align: 'center' });

  const dateStr = completed_at
    ? new Date(completed_at).toLocaleDateString()
    : new Date().toLocaleDateString();

  doc.setFontSize(12);
  doc.setTextColor(117, 117, 117);
  doc.text(`Date: ${dateStr}`, 100, 185, { align: 'center' });
  doc.text(`Certificate ID: ${cert_id}`, 200, 185, { align: 'center' });

  doc.save(`AgriLMS_Certificate_${course_title.replace(/\s+/g, '_')}.pdf`);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AgriLMS() {
  const SESSION_KEYS = {
    ACTIVE_COURSE: 'agri:lms:active-course',
    ACTIVE_LESSON: 'agri:lms:active-lesson',
  };

  const getSessionValue = (key) => {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const [activeCourse, setActiveCourse] = useState(() => {
  const stored = getSessionValue(SESSION_KEYS.ACTIVE_COURSE);

  if (!stored) return null;

  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
  });

  const [activeLesson, setActiveLesson] = useState(() => {
    const stored = getSessionValue(SESSION_KEYS.ACTIVE_LESSON);

    if (!stored) return null;

    try {
      return JSON.parse(stored);
    } catch {
      return null;
    }
  });

  const [showAdvisor, setShowAdvisor] = useState(false);

  // Server-authoritative progress: { [courseId]: { lessons: { [lessonId]: true }, completedAt } }
  const [serverProgress, setServerProgress] = useState({});
  const [progressLoading, setProgressLoading] = useState(true);
  const [progressError, setProgressError]     = useState(null);

  // Per-lesson "marking in progress" state to prevent double-clicks
  const [markingLesson, setMarkingLesson] = useState(null);

  // Per-course "fetching certificate" state
  const [fetchingCert, setFetchingCert] = useState(null);

  useEffect(() => {
    try {
      if (activeCourse?.id) {
        sessionStorage.setItem(
          SESSION_KEYS.ACTIVE_COURSE,
          JSON.stringify(activeCourse)
        );
      } else {
        sessionStorage.removeItem(SESSION_KEYS.ACTIVE_COURSE);
      }

      if (activeLesson?.id) {
        sessionStorage.setItem(
          SESSION_KEYS.ACTIVE_LESSON,
          JSON.stringify(activeLesson)
        );
      } else {
        sessionStorage.removeItem(SESSION_KEYS.ACTIVE_LESSON);
      }
    } catch (error) {
      console.warn("Unable to persist LMS session state");
    }
  }, [activeCourse, activeLesson]);

  // ---------------------------------------------------------------------------
  // Load server-side progress on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setProgressLoading(true);
    setProgressError(null);

    apiClient.get('/api/lms/progress')
      .then(res => {
        if (!cancelled) setServerProgress(res.data.progress ?? {});
      })
      .catch(err => {
        if (!cancelled) {
          const msg = err?.response?.status === 401
            ? 'Please log in to view your progress.'
            : 'Could not load progress. Please refresh.';
          setProgressError(msg);
        }
      })
      .finally(() => { if (!cancelled) setProgressLoading(false); });

    return () => { cancelled = true; };
  }, []);

  // ---------------------------------------------------------------------------
  // Mark a lesson complete — writes to the server, then updates local state
  // ---------------------------------------------------------------------------
  const markAsComplete = useCallback(async (lessonId) => {
    if (markingLesson === lessonId) return;
    setMarkingLesson(lessonId);
    try {
      const res = await apiClient.post('/api/lms/complete-lesson', { lesson_id: lessonId });
      const { course_id } = res.data;
      // Merge the newly completed lesson into local server-progress state.
      setServerProgress(prev => ({
        ...prev,
        [course_id]: {
          ...prev[course_id],
          lessons: {
            ...(prev[course_id]?.lessons ?? {}),
            [lessonId]: true,
          },
          // If the server says the course is now complete, record the timestamp.
          ...(res.data.course_complete && !prev[course_id]?.completedAt
            ? { completedAt: new Date().toISOString() }
            : {}),
        },
      }));
    } catch (err) {
      const status = err?.response?.status;
      if (status === 401) {
        alert('Please log in to save your progress.');
      } else {
        alert('Could not save progress. Please try again.');
      }
    } finally {
      setMarkingLesson(null);
    }
  }, [markingLesson]);

  // ---------------------------------------------------------------------------
  // Request certificate data from the server, then generate the PDF locally
  // ---------------------------------------------------------------------------
  const generateCertificate = useCallback(async (course) => {
    if (fetchingCert === course.id) return;
    setFetchingCert(course.id);
    try {
      const res = await apiClient.get(`/api/lms/certificate/${course.id}`);
      downloadCertificate(res.data.certificate);
    } catch (err) {
      const status = err?.response?.status;
      if (status === 403) {
        alert('Complete all lessons before downloading the certificate.');
      } else if (status === 401) {
        alert('Please log in to download your certificate.');
      } else {
        alert('Could not generate certificate. Please try again.');
      }
    } finally {
      setFetchingCert(null);
    }
  }, [fetchingCert]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  if (progressLoading) {
    return (
      <div className="lms-container" style={{ textAlign: 'center', padding: '4rem' }}>
        <Loader size={32} className="spin" />
        <p>Loading your progress…</p>
      </div>
    );
  }

  if (progressError) {
    return (
      <div className="lms-container" style={{ textAlign: 'center', padding: '4rem' }}>
        <p style={{ color: '#c62828' }}>{progressError}</p>
      </div>
    );
  }

  let content;

  if (activeCourse) {
    const pct = getCourseProgress(activeCourse, serverProgress);
    const lessonsDone = serverProgress[activeCourse.id]?.lessons ?? {};

    content = (
      <div className="lms-content active-course">
        <div className="lms-header active-header">
          <button className="back-btn" onClick={() => { setActiveCourse(null); setActiveLesson(null); }}>
            <ChevronRight style={{ transform: 'rotate(180deg)' }} /> Back to Courses
          </button>
          <div className="active-title-group">
            <h2>{activeCourse.title}</h2>
            <div className="course-progress-tag">{pct}% Completed</div>
          </div>
        </div>

        <div className="course-view-grid">
          <div className="video-section">
            {activeLesson ? (
              <>
                <div className="video-container">
                  <iframe
                    src={activeLesson.videoUrl}
                    title={activeLesson.title}
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
                <div className="video-info">
                  <div className="lesson-text">
                    <h3>{activeLesson.title}</h3>
                    <p className="instruction-text">
                      {lessonsDone[activeLesson.id]
                        ? "You've completed this lesson!"
                        : 'Watch the video and click the button to mark it as finished.'}
                    </p>
                  </div>
                  <div className="video-actions">
                    <button
                      className={`complete-btn ${lessonsDone[activeLesson.id] ? 'completed' : 'pulse'}`}
                      onClick={() => markAsComplete(activeLesson.id)}
                      disabled={!!lessonsDone[activeLesson.id] || markingLesson === activeLesson.id}
                    >
                      {markingLesson === activeLesson.id
                        ? <Loader size={18} className="spin" />
                        : lessonsDone[activeLesson.id]
                          ? <CheckCircle size={18} />
                          : null}
                      {lessonsDone[activeLesson.id] ? 'Completed' : 'Mark as Complete'}
                    </button>
                    {pct === 100 && (
                      <button
                        className="cert-btn-mini"
                        onClick={() => generateCertificate(activeCourse)}
                        disabled={fetchingCert === activeCourse.id}
                      >
                        {fetchingCert === activeCourse.id
                          ? <Loader size={16} className="spin" />
                          : <Award size={16} />}
                        {' '}Get Certificate
                      </button>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="video-placeholder">
                <Play size={48} />
                <p>Select a lesson to start learning</p>
              </div>
            )}
          </div>

          <div className="lessons-list">
            <h3>Course Content</h3>
            {activeCourse.lessons.map((lesson, idx) => (
              <div
                key={lesson.id}
                className={`lesson-item ${activeLesson?.id === lesson.id ? 'active' : ''} ${lessonsDone[lesson.id] ? 'finished' : ''}`}
                onClick={() => setActiveLesson(lesson)}
              >
                <div className="lesson-num">{idx + 1}</div>
                <div className="lesson-details">
                  <h4>{lesson.title}</h4>
                  <span><Clock size={12} /> {lesson.duration}</span>
                </div>
                {lessonsDone[lesson.id] && <CheckCircle size={16} className="status-icon" />}
              </div>
            ))}

            {pct === 100 && (
              <div className="certification-ready">
                <Award size={32} />
                <p>Congratulations! You've finished this course.</p>
                <button
                  className="cert-btn"
                  onClick={() => generateCertificate(activeCourse)}
                  disabled={fetchingCert === activeCourse.id}
                >
                  {fetchingCert === activeCourse.id
                    ? <Loader size={18} className="spin" />
                    : <Download size={18} />}
                  {' '}Download Certificate
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  } else {
    content = (
      <div className="lms-container">
        <div className="lms-header">
          <h1><BookOpen size={28} /> Agri-LMS Learning Portal</h1>
          <p>Empowering the next generation of farmers through digital education.</p>
        </div>

        <div className="course-categories">
          {COURSES.map(course => {
            const pct = getCourseProgress(course, serverProgress);
            return (
              <div key={course.id} className="course-card">
                <div className="course-badge">{course.category}</div>
                <h3>{course.title}</h3>
                <div className="course-meta">
                  <span><Clock size={14} /> {course.duration}</span>
                  <span><Play size={14} /> {course.lessons.length} Lessons</span>
                </div>

                <div className="progress-bar-container">
                  <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
                </div>
                <div className="progress-text">
                  {pct}% Completed
                  {pct === 100 && <CheckCircle size={14} style={{ color: '#4caf50', marginLeft: '8px' }} />}
                </div>

                {pct === 100 ? (
                  <div className="card-actions">
                    <button className="start-course-btn" onClick={() => setActiveCourse(course)}>
                      Review Course
                    </button>
                    <button
                      className="cert-btn-small"
                      onClick={e => { e.stopPropagation(); generateCertificate(course); }}
                      disabled={fetchingCert === course.id}
                    >
                      {fetchingCert === course.id
                        ? <Loader size={16} className="spin" />
                        : <Award size={16} />}
                      {' '}Get Certificate
                    </button>
                  </div>
                ) : (
                  <button className="start-course-btn" onClick={() => setActiveCourse(course)}>
                    {pct > 0 ? 'Continue Learning' : 'Start Course'}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <>
      {content}
      <button className="advisor-fab" onClick={() => setShowAdvisor(true)} aria-label="Open AI Advisor">
        <MessageCircle size={24} />
      </button>
      {showAdvisor && (
        <div className="advisor-overlay" onClick={() => setShowAdvisor(false)}>
          <div className="advisor-modal" onClick={e => e.stopPropagation()}>
            <SoilChatbot onClose={() => setShowAdvisor(false)} />
          </div>
        </div>
      )}
    </>
  );
}
