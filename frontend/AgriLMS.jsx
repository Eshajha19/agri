import React, { useState, useEffect } from 'react';
import './AgriLMS.css';
import { Play, CheckCircle, Award, BookOpen, Clock, Download, ChevronRight, MessageCircle } from 'lucide-react';
import jsPDF from 'jspdf';
import SoilChatbot from './SoilChatbot';

const COURSES = [
  {
    id: 'soil-health',
    title: 'Advanced Soil Management',
    category: 'Soil',
    duration: '45 mins',
    lessons: [
      { id: 's1', title: 'Testing Soil pH at Home', duration: '10:00', videoUrl: 'https://www.youtube.com/embed/5_gYbLGiVMI' },
      { id: 's2', title: 'Organic Matter Enrichment', duration: '15:00', videoUrl: 'https://www.youtube.com/embed/elEuxFzbTO0' },
      { id: 's3', title: 'Crop Rotation Basics', duration: '20:00', videoUrl: 'https://www.youtube.com/embed/3QLYFg4NIN8' }
    ]
  },
  {
    id: 'pest-control',
    title: 'Organic Pest Management',
    category: 'Pest Control',
    duration: '30 mins',
    lessons: [
      { id: 'p1', title: 'Natural Insecticides', duration: '12:00', videoUrl: 'https://www.youtube.com/embed/ZyvcmpyD7FM' },
      { id: 'p2', title: 'Biological Control Agents', duration: '18:00', videoUrl: 'https://www.youtube.com/embed/g6LMw9I6rxU' }
    ]
  },
  {
    id: 'modern-tools',
    title: 'Drones in Agriculture',
    category: 'Technology',
    duration: '25 mins',
    lessons: [
      { id: 't1', title: 'Drone Mapping Basics', duration: '10:00', videoUrl: 'https://www.youtube.com/embed/QtXhHZP5SSY' },
      { id: 't2', title: 'Precision Spraying', duration: '15:00', videoUrl: 'https://www.youtube.com/embed/-0rAAqVeCG8' }
    ]
  }
];

export default function AgriLMS() {
  const [activeCourse, setActiveCourse] = useState(null);
  const [activeLesson, setActiveLesson] = useState(null);
  const [progress, setProgress] = useState(() => {
    const saved = localStorage.getItem('agriLmsProgress');
    return saved ? JSON.parse(saved) : {};
  });
  const [showAdvisor, setShowAdvisor] = useState(false);

  useEffect(() => {
    localStorage.setItem('agriLmsProgress', JSON.stringify(progress));
  }, [progress]);

  const markAsComplete = (lessonId) => {
    setProgress(prev => ({ ...prev, [lessonId]: true }));
  };

  const getCourseProgress = (course) => {
    const completed = course.lessons.filter(l => progress[l.id]).length;
    return Math.round((completed / course.lessons.length) * 100);
  };

  const generateCertificate = (course) => {
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
    doc.text('Fasal Saathi Student', 148.5, 105, { align: 'center' });

    doc.setFontSize(20);
    doc.setFont('helvetica', 'normal');
    doc.text('has successfully completed the course', 148.5, 130, { align: 'center' });

    doc.setFontSize(25);
    doc.setTextColor(46, 125, 50);
    doc.text(course.title, 148.5, 155, { align: 'center' });

    doc.setFontSize(15);
    doc.setTextColor(117, 117, 117);
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 148.5, 185, { align: 'center' });

    doc.save(`AgriLMS_Certificate_${course.id}.pdf`);
  };

  let content;

  if (activeCourse) {
    content = (
      <div className="lms-content active-course">
        <div className="lms-header active-header">
          <button className="back-btn" onClick={() => { setActiveCourse(null); setActiveLesson(null); }}>
            <ChevronRight style={{ transform: 'rotate(180deg)' }} /> Back to Courses
          </button>
          <div className="active-title-group">
            <h2>{activeCourse.title}</h2>
            <div className="course-progress-tag">
              {getCourseProgress(activeCourse)}% Completed
            </div>
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
                  ></iframe>
                </div>
                <div className="video-info">
                  <div className="lesson-text">
                    <h3>{activeLesson.title}</h3>
                    <p className="instruction-text">
                      {progress[activeLesson.id]
                        ? "You've completed this lesson!"
                        : "Watch the video and click the button to mark it as finished."}
                    </p>
                  </div>
                  <div className="video-actions">
                    <button
                      className={`complete-btn ${progress[activeLesson.id] ? 'completed' : 'pulse'}`}
                      onClick={() => markAsComplete(activeLesson.id)}
                    >
                      {progress[activeLesson.id] ? <CheckCircle size={18} /> : null}
                      {progress[activeLesson.id] ? 'Completed' : 'Mark as Complete'}
                    </button>
                    {getCourseProgress(activeCourse) === 100 && (
                      <button className="cert-btn-mini" onClick={() => generateCertificate(activeCourse)}>
                        <Award size={16} /> Get Certificate
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
                className={`lesson-item ${activeLesson?.id === lesson.id ? 'active' : ''} ${progress[lesson.id] ? 'finished' : ''}`}
                onClick={() => setActiveLesson(lesson)}
              >
                <div className="lesson-num">{idx + 1}</div>
                <div className="lesson-details">
                  <h4>{lesson.title}</h4>
                  <span><Clock size={12} /> {lesson.duration}</span>
                </div>
                {progress[lesson.id] && <CheckCircle size={16} className="status-icon" />}
              </div>
            ))}

            {getCourseProgress(activeCourse) === 100 && (
              <div className="certification-ready">
                <Award size={32} />
                <p>Congratulations! You've finished this course.</p>
                <button className="cert-btn" onClick={() => generateCertificate(activeCourse)}>
                  <Download size={18} /> Download Certificate
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
          {COURSES.map(course => (
            <div key={course.id} className="course-card">
              <div className="course-badge">{course.category}</div>
              <h3>{course.title}</h3>
              <div className="course-meta">
                <span><Clock size={14} /> {course.duration}</span>
                <span><Play size={14} /> {course.lessons.length} Lessons</span>
              </div>

              <div className="progress-bar-container">
                <div className="progress-bar-fill" style={{ width: `${getCourseProgress(course)}%` }}></div>
              </div>
              <div className="progress-text">
                {getCourseProgress(course)}% Completed
                {getCourseProgress(course) === 100 && <CheckCircle size={14} style={{ color: '#4caf50', marginLeft: '8px' }} />}
              </div>

              {getCourseProgress(course) === 100 ? (
                <div className="card-actions">
                  <button className="start-course-btn" onClick={() => setActiveCourse(course)}>
                    Review Course
                  </button>
                  <button className="cert-btn-small" onClick={(e) => { e.stopPropagation(); generateCertificate(course); }}>
                    <Award size={16} /> Get Certificate
                  </button>
                </div>
              ) : (
                <button className="start-course-btn" onClick={() => setActiveCourse(course)}>
                  {getCourseProgress(course) > 0 ? 'Continue Learning' : 'Start Course'}
                </button>
              )}
            </div>
          ))}
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
