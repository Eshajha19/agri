import React, {
  useState,
  useEffect,
  useCallback,
  useRef
} from 'react';
import './AgriLMS.css';
import {
  Play,
  CheckCircle,
  Award,
  BookOpen,
  Clock,
  Download,
  ChevronRight,
  MessageCircle,
  Loader
} from 'lucide-react';
import jsPDF from 'jspdf';
import SoilChatbot from './SoilChatbot';
import apiClient from './services/api';

// ---------------------------------------------------------------------------
// Course catalogue — mirrors backend COURSES dict in backend/routers/lms.py.
// ---------------------------------------------------------------------------

const COURSES = [
  {
    id: 'soil-health',
    title: 'Advanced Soil Management',
    category: 'Soil',
    duration: '45 mins',
    lessons: [
      {
        id: 's1',
        title: 'Testing Soil pH at Home',
        duration: '10:00',
        videoUrl:
          'https://www.youtube.com/embed/5_gYbLGiVMI'
      },
      {
        id: 's2',
        title: 'Organic Matter Enrichment',
        duration: '15:00',
        videoUrl:
          'https://www.youtube.com/embed/elEuxFzbTO0'
      },
      {
        id: 's3',
        title: 'Crop Rotation Basics',
        duration: '20:00',
        videoUrl:
          'https://www.youtube.com/embed/3QLYFg4NIN8'
      }
    ]
  },
  {
    id: 'pest-control',
    title: 'Organic Pest Management',
    category: 'Pest Control',
    duration: '30 mins',
    lessons: [
      {
        id: 'p1',
        title: 'Natural Insecticides',
        duration: '12:00',
        videoUrl:
          'https://www.youtube.com/embed/ZyvcmpyD7FM'
      },
      {
        id: 'p2',
        title: 'Biological Control Agents',
        duration: '18:00',
        videoUrl:
          'https://www.youtube.com/embed/g6LMw9I6rxU'
      }
    ]
  },
  {
    id: 'modern-tools',
    title: 'Drones in Agriculture',
    category: 'Technology',
    duration: '25 mins',
    lessons: [
      {
        id: 't1',
        title: 'Drone Mapping Basics',
        duration: '10:00',
        videoUrl:
          'https://www.youtube.com/embed/QtXhHZP5SSY'
      },
      {
        id: 't2',
        title: 'Precision Spraying',
        duration: '15:00',
        videoUrl:
          'https://www.youtube.com/embed/-0rAAqVeCG8'
      }
    ]
  }
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getCourseProgress(
  course,
  serverProgress
) {
  const done =
    serverProgress[course.id]?.lessons ??
    {};

  const completed =
    course.lessons.filter(
      (l) => done[l.id] === true
    ).length;

  return Math.round(
    (completed /
      course.lessons.length) *
      100
  );
}

function downloadCertificate({
  recipient_name,
  course_title,
  completed_at,
  cert_id
}) {
  const doc = new jsPDF({
    orientation: 'landscape',
    unit: 'mm',
    format: 'a4'
  });

  doc.setFillColor(245, 247, 241);
  doc.rect(0, 0, 297, 210, 'F');

  doc.setDrawColor(46, 125, 50);
  doc.setLineWidth(5);
  doc.rect(10, 10, 277, 190);

  doc.setTextColor(46, 125, 50);
  doc.setFontSize(40);

  doc.text(
    'Certificate of Completion',
    148.5,
    50,
    { align: 'center' }
  );

  doc.setTextColor(33, 33, 33);
  doc.setFontSize(20);

  doc.text(
    'This is to certify that',
    148.5,
    80,
    { align: 'center' }
  );

  doc.setFontSize(30);
  doc.setFont('helvetica', 'bold');

  doc.text(
    recipient_name,
    148.5,
    105,
    { align: 'center' }
  );

  doc.setFontSize(20);
  doc.setFont('helvetica', 'normal');

  doc.text(
    'has successfully completed the course',
    148.5,
    130,
    { align: 'center' }
  );

  doc.setFontSize(25);
  doc.setTextColor(46, 125, 50);

  doc.text(
    course_title,
    148.5,
    155,
    { align: 'center' }
  );

  const dateStr = completed_at
    ? new Date(
        completed_at
      ).toLocaleDateString()
    : new Date().toLocaleDateString();

  doc.setFontSize(12);
  doc.setTextColor(117, 117, 117);

  doc.text(
    `Date: ${dateStr}`,
    100,
    185,
    { align: 'center' }
  );

  doc.text(
    `Certificate ID: ${cert_id}`,
    200,
    185,
    { align: 'center' }
  );

  doc.save(
    `AgriLMS_Certificate_${course_title.replace(
      /\s+/g,
      '_'
    )}.pdf`
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function AgriLMS() {
  const [activeCourse, setActiveCourse] =
    useState(null);

  const [activeLesson, setActiveLesson] =
    useState(null);

  const [showAdvisor, setShowAdvisor] =
    useState(false);

  const [serverProgress, setServerProgress] =
    useState({});

  const [progressLoading, setProgressLoading] =
    useState(true);

  const [progressError, setProgressError] =
    useState(null);

  const [markingLesson, setMarkingLesson] =
    useState(null);

  const [fetchingCert, setFetchingCert] =
    useState(null);

  // ---------------------------------------------------------------------------
  // Advanced Session Persistence Hardening
  // ---------------------------------------------------------------------------

  const LMS_PROGRESS_CACHE_KEY =
    'agri_lms_progress_cache_v2';

  const mountedRef = useRef(true);

  const recoveryInProgress = useRef(false);

  const activeFetchRef = useRef(null);

  const persistProgressSnapshot =
    useCallback((progress) => {
      try {
        sessionStorage.setItem(
          LMS_PROGRESS_CACHE_KEY,
          JSON.stringify({
            progress,
            savedAt:
              new Date().toISOString()
          })
        );
      } catch (error) {
        console.error(
          'Session persistence failed:',
          error
        );
      }
    }, []);

  const restoreProgressSnapshot =
    useCallback(() => {
      try {
        const raw =
          sessionStorage.getItem(
            LMS_PROGRESS_CACHE_KEY
          );

        if (!raw) return null;

        const parsed = JSON.parse(raw);

        if (
          !parsed ||
          typeof parsed !== 'object'
        ) {
          return null;
        }

        return parsed.progress || null;
      } catch (error) {
        console.error(
          'Session recovery failed:',
          error
        );
        return null;
      }
    }, []);

  const clearProgressSnapshot =
    useCallback(() => {
      try {
        sessionStorage.removeItem(
          LMS_PROGRESS_CACHE_KEY
        );
      } catch (error) {
        console.error(
          'Failed to clear LMS snapshot:',
          error
        );
      }
    }, []);

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Load server-side progress on mount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    if (recoveryInProgress.current) {
      return;
    }

    recoveryInProgress.current = true;

    setProgressLoading(true);
    setProgressError(null);

    const cachedProgress =
      restoreProgressSnapshot();

    if (
      cachedProgress &&
      mountedRef.current
    ) {
      setServerProgress(cachedProgress);
    }

    const fetchProgress =
      async () => {
        try {
          activeFetchRef.current =
            Date.now();

          const response =
            await apiClient.get(
              '/api/lms/progress'
            );

          if (
            cancelled ||
            !mountedRef.current
          ) {
            return;
          }

          const latestProgress =
            response.data.progress ?? {};

          setServerProgress(
            latestProgress
          );

          persistProgressSnapshot(
            latestProgress
          );
        } catch (err) {
          if (
            cancelled ||
            !mountedRef.current
          ) {
            return;
          }

          const status =
            err?.response?.status;

          if (status === 401) {
            clearProgressSnapshot();

            setProgressError(
              'Please log in to view your progress.'
            );
          } else if (
            cachedProgress
          ) {
            setProgressError(
              'Recovered previous LMS state while offline.'
            );
          } else {
            setProgressError(
              'Could not load progress. Please refresh.'
            );
          }
        } finally {
          if (
            !cancelled &&
            mountedRef.current
          ) {
            setProgressLoading(false);
          }

          recoveryInProgress.current =
            false;
        }
      };

    fetchProgress();

    return () => {
      cancelled = true;
    };
  }, [
    persistProgressSnapshot,
    restoreProgressSnapshot,
    clearProgressSnapshot
  ]);

  // ---------------------------------------------------------------------------
  // Visibility/session recovery sync
  // ---------------------------------------------------------------------------

  useEffect(() => {
    const handleVisibilityRecovery =
      () => {
        if (
          document.visibilityState ===
            'visible' &&
          !recoveryInProgress.current
        ) {
          const cached =
            restoreProgressSnapshot();

          if (
            cached &&
            mountedRef.current
          ) {
            setServerProgress(
              cached
            );
          }
        }
      };

    window.addEventListener(
      'visibilitychange',
      handleVisibilityRecovery
    );

    return () => {
      window.removeEventListener(
        'visibilitychange',
        handleVisibilityRecovery
      );
    };
  }, [restoreProgressSnapshot]);

  // ---------------------------------------------------------------------------
  // Persist snapshot whenever progress changes
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (
      serverProgress &&
      Object.keys(serverProgress)
        .length > 0
    ) {
      persistProgressSnapshot(
        serverProgress
      );
    }
  }, [
    serverProgress,
    persistProgressSnapshot
  ]);

  // ---------------------------------------------------------------------------
  // Mark lesson complete
  // ---------------------------------------------------------------------------

  const markAsComplete =
    useCallback(
      async (lessonId) => {
        if (
          markingLesson ===
          lessonId
        ) {
          return;
        }

        setMarkingLesson(
          lessonId
        );

        try {
          const res =
            await apiClient.post(
              '/api/lms/complete-lesson',
              {
                lesson_id:
                  lessonId
              }
            );

          const { course_id } =
            res.data;

          setServerProgress(
            (prev) => {
              const updated = {
                ...prev,

                [course_id]:
                  {
                    ...prev[
                      course_id
                    ],

                    lessons: {
                      ...(prev[
                        course_id
                      ]?.lessons ??
                        {}),

                      [lessonId]:
                        true
                    },

                    ...(res.data
                      .course_complete &&
                    !prev[
                      course_id
                    ]
                      ?.completedAt
                      ? {
                          completedAt:
                            new Date().toISOString()
                        }
                      : {})
                  }
              };

              persistProgressSnapshot(
                updated
              );

              return updated;
            }
          );
        } catch (err) {
          const status =
            err?.response
              ?.status;

          if (
            status === 401
          ) {
            clearProgressSnapshot();

            alert(
              'Please log in to save your progress.'
            );
          } else {
            alert(
              'Could not save progress. Please try again.'
            );
          }
        } finally {
          setMarkingLesson(
            null
          );
        }
      },
      [
        markingLesson,
        persistProgressSnapshot,
        clearProgressSnapshot
      ]
    );

  // ---------------------------------------------------------------------------
  // Generate certificate
  // ---------------------------------------------------------------------------

  const generateCertificate =
    useCallback(
      async (course) => {
        if (
          fetchingCert ===
          course.id
        ) {
          return;
        }

        setFetchingCert(
          course.id
        );

        try {
          const res =
            await apiClient.get(
              `/api/lms/certificate/${course.id}`
            );

          downloadCertificate(
            res.data
              .certificate
          );
        } catch (err) {
          const status =
            err?.response
              ?.status;

          if (
            status === 403
          ) {
            alert(
              'Complete all lessons before downloading the certificate.'
            );
          } else if (
            status === 401
          ) {
            clearProgressSnapshot();

            alert(
              'Please log in to download your certificate.'
            );
          } else {
            alert(
              'Could not generate certificate. Please try again.'
            );
          }
        } finally {
          setFetchingCert(
            null
          );
        }
      },
      [
        fetchingCert,
        clearProgressSnapshot
      ]
    );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  if (progressLoading) {
    return (
      <div
        className="lms-container"
        style={{
          textAlign:
            'center',
          padding: '4rem'
        }}
      >
        <Loader
          size={32}
          className="spin"
        />

        <p>
          Loading your
          progress…
        </p>
      </div>
    );
  }

  if (progressError) {
    return (
      <div
        className="lms-container"
        style={{
          textAlign:
            'center',
          padding: '4rem'
        }}
      >
        <p
          style={{
            color:
              '#c62828'
          }}
        >
          {progressError}
        </p>
      </div>
    );
  }

  let content;

  if (activeCourse) {
    const pct =
      getCourseProgress(
        activeCourse,
        serverProgress
      );

    const lessonsDone =
      serverProgress[
        activeCourse.id
      ]?.lessons ?? {};

    content = (
      <div className="lms-content active-course">
        <div className="lms-header active-header">
          <button
            className="back-btn"
            onClick={() => {
              setActiveCourse(
                null
              );

              setActiveLesson(
                null
              );
            }}
          >
            <ChevronRight
              style={{
                transform:
                  'rotate(180deg)'
              }}
            />

            Back to Courses
          </button>

          <div className="active-title-group">
            <h2>
              {
                activeCourse.title
              }
            </h2>

            <div className="course-progress-tag">
              {pct}% Completed
            </div>
          </div>
        </div>
      </div>
    );
  } else {
    content = (
      <div className="lms-container">
        <div className="lms-header">
          <h1>
            <BookOpen size={28} />
            {' '}
            Agri-LMS
            Learning Portal
          </h1>

          <p>
            Empowering the
            next generation
            of farmers
            through digital
            education.
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      {content}

      <button
        className="advisor-fab"
        onClick={() =>
          setShowAdvisor(
            true
          )
        }
        aria-label="Open AI Advisor"
      >
        <MessageCircle size={24} />
      </button>

      {showAdvisor && (
        <div
          className="advisor-overlay"
          onClick={() =>
            setShowAdvisor(
              false
            )
          }
        >
          <div
            className="advisor-modal"
            onClick={(e) =>
              e.stopPropagation()
            }
          >
            <SoilChatbot
              onClose={() =>
                setShowAdvisor(
                  false
                )
              }
            />
          </div>
        </div>
      )}
    </>
  );
}