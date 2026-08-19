import emailjs from "@emailjs/browser";

// ============================================================
// EmailJS Configuration
// ============================================================

const PUBLIC_KEY = import.meta.env.VITE_EMAILJS_PUBLIC_KEY;
const SERVICE_ID = import.meta.env.VITE_EMAILJS_SERVICE_ID;
const TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID;

// ============================================================
// Activity Configuration
// ============================================================

const ACTIVITY_TYPE_LABELS = {
  sowing: "Sowing",
  irrigation: "Irrigation",
  spraying: "Spraying",
  harvest: "Harvest",
  other: "Other",
};

const ACTIVITY_TYPE_COLORS = {
  sowing: "#10b981",
  irrigation: "#3b82f6",
  spraying: "#f59e0b",
  harvest: "#8b5cf6",
  other: "#6b7280",
};

// ============================================================
// Utility Functions
// ============================================================

const escapeHtml = (value) => {
  if (value === null || value === undefined) {
    return "";
  }

  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
};

const formatDateLabel = (dateStr) => {
  try {
    const date = new Date(dateStr);

    if (Number.isNaN(date.getTime())) {
      return dateStr || "";
    }

    return date.toLocaleDateString("en-IN", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return dateStr || "";
  }
};

const formatTimeLabel = (timeStr) => {
  if (!timeStr) {
    return "";
  }

  try {
    const [hours, minutes] = String(timeStr).split(":").map(Number);

    if (
      Number.isNaN(hours) ||
      Number.isNaN(minutes)
    ) {
      return timeStr;
    }

    const date = new Date();

    date.setHours(hours, minutes, 0, 0);

    return date.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  } catch {
    return timeStr;
  }
};

const isSameCalendarDay = (dateA, dateB) => {
  if (!dateA || !dateB) {
    return false;
  }

  const a = new Date(dateA);
  const b = new Date(dateB);

  if (
    Number.isNaN(a.getTime()) ||
    Number.isNaN(b.getTime())
  ) {
    return false;
  }

  return (
    a.getDate() === b.getDate() &&
    a.getMonth() === b.getMonth() &&
    a.getFullYear() === b.getFullYear()
  );
};

const isFutureActivity = (dateStr) => {
  if (!dateStr) {
    return false;
  }

  const activityDate = new Date(dateStr);

  if (Number.isNaN(activityDate.getTime())) {
    return false;
  }

  const today = new Date();

  today.setHours(0, 0, 0, 0);

  return activityDate > today;
};

// ============================================================
// HTML Email Builder
// ============================================================

export const buildDailyReminderHtml = ({
  userName = "Farmer",
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  const today = new Date();

  const todayLabel = formatDateLabel(
    today.toISOString()
  );

  // ----------------------------------------------------------
  // Today's activities
  // ----------------------------------------------------------

  const todayActivities = activities.filter(
    (activity) => {
      if (!activity?.date) {
        return false;
      }

      return isSameCalendarDay(
        activity.date,
        today
      );
    }
  );

  const sortedToday = [...todayActivities].sort(
    (a, b) => {
      const aTime = a?.time || "00:00";
      const bTime = b?.time || "00:00";

      return aTime.localeCompare(bTime);
    }
  );

  // ----------------------------------------------------------
  // Upcoming reminders
  // ----------------------------------------------------------

  const upcoming = [...upcomingReminders]
    .filter((activity) =>
      isFutureActivity(activity?.date)
    )
    .sort((a, b) => {
      const aDate = `${a?.date || ""}T${
        a?.time || "00:00"
      }`;

      const bDate = `${b?.date || ""}T${
        b?.time || "00:00"
      }`;

      return aDate.localeCompare(bDate);
    })
    .slice(0, 5);

  // ----------------------------------------------------------
  // Today's activity row
  // ----------------------------------------------------------

  const renderActivityRow = (activity) => {
    const typeLabel =
      ACTIVITY_TYPE_LABELS[activity?.type] ||
      "Other";

    const color =
      ACTIVITY_TYPE_COLORS[activity?.type] ||
      "#6b7280";

    const timeLabel = formatTimeLabel(
      activity?.time
    );

    const description = escapeHtml(
      activity?.description || ""
    );

    const title = escapeHtml(
      activity?.title || "Farm activity"
    );

    const cropType = escapeHtml(
      activity?.cropType || ""
    );

    return `
      <tr>
        <td
          style="
            padding: 12px 16px;
            border-bottom: 1px solid #f0fdf4;
            vertical-align: top;
          "
        >

          <div
            style="
              display: flex;
              align-items: center;
              gap: 10px;
            "
          >

            <span
              style="
                display: inline-flex;
                align-items: center;
                background: ${color}18;
                color: ${color};
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                white-space: nowrap;
              "
            >
              ${escapeHtml(typeLabel)}
            </span>

            ${
              timeLabel
                ? `
                  <span
                    style="
                      color: #6b7280;
                      font-size: 13px;
                    "
                  >
                    ${escapeHtml(timeLabel)}
                  </span>
                `
                : ""
            }

          </div>

          <div
            style="
              margin-top: 6px;
              font-weight: 600;
              color: #111827;
              font-size: 15px;
            "
          >
            ${title}
          </div>

          ${
            description
              ? `
                <div
                  style="
                    margin-top: 4px;
                    color: #4b5563;
                    font-size: 13px;
                    line-height: 1.5;
                  "
                >
                  ${description}
                </div>
              `
              : ""
          }

          ${
            cropType
              ? `
                <div
                  style="
                    margin-top: 4px;
                    color: #059669;
                    font-size: 12px;
                    font-weight: 500;
                  "
                >
                  Crop: ${cropType}
                </div>
              `
              : ""
          }

        </td>
      </tr>
    `;
  };

  // ----------------------------------------------------------
  // Upcoming reminder row
  // ----------------------------------------------------------

  const renderUpcomingRow = (activity) => {
    const typeLabel =
      ACTIVITY_TYPE_LABELS[activity?.type] ||
      "Other";

    const color =
      ACTIVITY_TYPE_COLORS[activity?.type] ||
      "#6b7280";

    const dateLabel = formatDateLabel(
      activity?.date
    );

    const timeLabel = formatTimeLabel(
      activity?.time
    );

    const title = escapeHtml(
      activity?.title || "Farm activity"
    );

    return `
      <tr>
        <td
          style="
            padding: 10px 16px;
            border-bottom: 1px solid #f3f4f6;
            vertical-align: top;
          "
        >

          <div
            style="
              display: flex;
              align-items: center;
              gap: 8px;
              flex-wrap: wrap;
            "
          >

            <span
              style="
                display: inline-flex;
                align-items: center;
                background: ${color}18;
                color: ${color};
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
              "
            >
              ${escapeHtml(typeLabel)}
            </span>

            <span
              style="
                font-weight: 500;
                color: #111827;
                font-size: 14px;
              "
            >
              ${title}
            </span>

            <span
              style="
                color: #6b7280;
                font-size: 12px;
              "
            >
              ${escapeHtml(dateLabel)}
              ${
                timeLabel
                  ? ` at ${escapeHtml(timeLabel)}`
                  : ""
              }
            </span>

          </div>

        </td>
      </tr>
    `;
  };

  // ----------------------------------------------------------
  // Build rows
  // ----------------------------------------------------------

  const todayRows =
    sortedToday.length > 0
      ? sortedToday
          .map(renderActivityRow)
          .join("")
      : `
        <tr>
          <td
            style="
              padding: 20px 16px;
              text-align: center;
              color: #9ca3af;
              font-size: 14px;
            "
          >
            No activities scheduled for today.
          </td>
        </tr>
      `;

  const upcomingRows =
    upcoming.length > 0
      ? upcoming
          .map(renderUpcomingRow)
          .join("")
      : `
        <tr>
          <td
            style="
              padding: 16px;
              text-align: center;
              color: #9ca3af;
              font-size: 14px;
            "
          >
            No upcoming reminders in the next few days.
          </td>
        </tr>
      `;

  // ----------------------------------------------------------
  // Complete HTML email
  // ----------------------------------------------------------

  return `
<!DOCTYPE html>

<html lang="en">

<head>

  <meta charset="UTF-8" />

  <meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
  />

  <title>
    Daily Farm Reminders
  </title>

</head>

<body
  style="
    margin: 0;
    padding: 0;
    background-color: #f9fafb;
    font-family:
      'Segoe UI',
      Roboto,
      Helvetica,
      Arial,
      sans-serif;
  "
>

<table
  role="presentation"
  width="100%"
  cellpadding="0"
  cellspacing="0"
  style="
    background-color: #f9fafb;
    padding: 24px 0;
  "
>

<tr>

<td align="center">

<table
  role="presentation"
  width="600"
  cellpadding="0"
  cellspacing="0"
  style="
    background-color: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    max-width: 600px;
    width: 100%;
  "
>

<!-- HEADER -->

<tr>

<td
  style="
    background:
      linear-gradient(
        135deg,
        #059669 0%,
        #10b981 100%
      );
    padding: 28px 32px;
    text-align: center;
  "
>

<h1
  style="
    margin: 0;
    color: #ffffff;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 0.3px;
  "
>
  ${escapeHtml(appName)}
</h1>

<p
  style="
    margin: 8px 0 0;
    color: #d1fae5;
    font-size: 14px;
    font-weight: 500;
  "
>
  Daily Farm Reminder
</p>

</td>

</tr>

<!-- GREETING -->

<tr>

<td
  style="
    padding: 28px 32px 12px;
  "
>

<p
  style="
    margin: 0;
    color: #111827;
    font-size: 16px;
    font-weight: 600;
  "
>
  Hello, ${escapeHtml(userName)}!
</p>

<p
  style="
    margin: 6px 0 0;
    color: #4b5563;
    font-size: 14px;
    line-height: 1.6;
  "
>
  Here is your farm schedule for
  <strong>${escapeHtml(todayLabel)}</strong>.
  Stay on top of your activities and
  keep your crops thriving.
</p>

</td>

</tr>

<!-- TODAY -->

<tr>

<td
  style="
    padding: 0 32px;
  "
>

<table
  role="presentation"
  width="100%"
  cellpadding="0"
  cellspacing="0"
  style="
    background-color: #f9fafb;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
  "
>

<tr>

<td
  style="
    padding: 14px 16px;
    background-color: #ecfdf5;
    border-bottom: 1px solid #d1fae5;
  "
>

<span
  style="
    color: #065f46;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  "
>
  Today's Activities
</span>

</td>

</tr>

${todayRows}

</table>

</td>

</tr>

<!-- UPCOMING -->

<tr>

<td
  style="
    padding: 20px 32px 0;
  "
>

<table
  role="presentation"
  width="100%"
  cellpadding="0"
  cellspacing="0"
  style="
    background-color: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
  "
>

<tr>

<td
  style="
    padding: 14px 16px;
    background-color: #eff6ff;
    border-bottom: 1px solid #dbeafe;
  "
>

<span
  style="
    color: #1e40af;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  "
>
  Upcoming Reminders
</span>

</td>

</tr>

${upcomingRows}

</table>

</td>

</tr>

<!-- FOOTER -->

<tr>

<td
  style="
    padding: 24px 32px 28px;
    text-align: center;
  "
>

<p
  style="
    margin: 0;
    color: #9ca3af;
    font-size: 12px;
    line-height: 1.6;
  "
>
  You are receiving this email because
  you have active reminders in
  ${escapeHtml(appName)}.
</p>

<p
  style="
    margin: 8px 0 0;
    color: #9ca3af;
    font-size: 12px;
  "
>
  &copy; ${new Date().getFullYear()}
  ${escapeHtml(appName)}.
  All rights reserved.
</p>

</td>

</tr>

</table>

</td>

</tr>

</table>

</body>

</html>
  `;
};

// ============================================================
// Plain Text Email Builder
// ============================================================

export const buildDailyReminderText = ({
  userName = "Farmer",
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  const today = new Date();

  const todayLabel = formatDateLabel(
    today.toISOString()
  );

  // ----------------------------------------------------------
  // Today's activities
  // ----------------------------------------------------------

  const todayActivities = activities.filter(
    (activity) => {
      if (!activity?.date) {
        return false;
      }

      return isSameCalendarDay(
        activity.date,
        today
      );
    }
  );

  const sortedToday = [...todayActivities].sort(
    (a, b) => {
      const aTime = a?.time || "00:00";
      const bTime = b?.time || "00:00";

      return aTime.localeCompare(bTime);
    }
  );

  // ----------------------------------------------------------
  // Upcoming
  // ----------------------------------------------------------

  const upcoming = [...upcomingReminders]
    .filter((activity) =>
      isFutureActivity(activity?.date)
    )
    .sort((a, b) => {
      const aDate = `${a?.date || ""}T${
        a?.time || "00:00"
      }`;

      const bDate = `${b?.date || ""}T${
        b?.time || "00:00"
      }`;

      return aDate.localeCompare(bDate);
    })
    .slice(0, 5);

  const lines = [
    `Hello ${userName}!`,
    "",
    `Here is your farm schedule for ${todayLabel}:`,
    "",
    "TODAY'S ACTIVITIES",
    "------------------",
  ];

  if (sortedToday.length === 0) {
    lines.push(
      "No activities scheduled for today."
    );
  } else {
    sortedToday.forEach((activity) => {
      const typeLabel =
        ACTIVITY_TYPE_LABELS[activity?.type] ||
        "Other";

      const timeLabel = formatTimeLabel(
        activity?.time
      );

      lines.push(
        `${timeLabel ? `[${timeLabel}] ` : ""}` +
          `${typeLabel}: ${
            activity?.title || "Farm activity"
          }`
      );

      if (activity?.description) {
        lines.push(
          `   ${activity.description}`
        );
      }

      if (activity?.cropType) {
        lines.push(
          `   Crop: ${activity.cropType}`
        );
      }

      lines.push("");
    });
  }

  lines.push("UPCOMING REMINDERS");
  lines.push("------------------");

  if (upcoming.length === 0) {
    lines.push(
      "No upcoming reminders in the next few days."
    );
  } else {
    upcoming.forEach((activity) => {
      const typeLabel =
        ACTIVITY_TYPE_LABELS[activity?.type] ||
        "Other";

      const dateLabel = formatDateLabel(
        activity?.date
      );

      const timeLabel = formatTimeLabel(
        activity?.time
      );

      lines.push(
        `${typeLabel}: ${
          activity?.title || "Farm activity"
        }`
      );

      lines.push(
        `   ${dateLabel}${
          timeLabel
            ? ` at ${timeLabel}`
            : ""
        }`
      );

      lines.push("");
    });
  }

  lines.push("---");

  lines.push(
    `You are receiving this email because you have active reminders in ${appName}.`
  );

  lines.push(
    `© ${today.getFullYear()} ${appName}. All rights reserved.`
  );

  return lines.join("\n");
};

// ============================================================
// Send Daily Reminder Email
// ============================================================

export const sendDailyReminderEmail = async ({
  toEmail,
  userName,
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  // ----------------------------------------------------------
  // Validate EmailJS configuration
  // ----------------------------------------------------------

  if (
    !SERVICE_ID ||
    !TEMPLATE_ID ||
    !PUBLIC_KEY
  ) {
    const missing = [];

    if (!SERVICE_ID) {
      missing.push(
        "VITE_EMAILJS_SERVICE_ID"
      );
    }

    if (!TEMPLATE_ID) {
      missing.push(
        "VITE_EMAILJS_TEMPLATE_ID"
      );
    }

    if (!PUBLIC_KEY) {
      missing.push(
        "VITE_EMAILJS_PUBLIC_KEY"
      );
    }

    throw new Error(
      `EmailJS is not configured. Missing: ${missing.join(
        ", "
      )}`
    );
  }

  // ----------------------------------------------------------
  // Validate recipient
  // ----------------------------------------------------------

  if (
    !toEmail ||
    !String(toEmail).trim()
  ) {
    throw new Error(
      "Recipient email is required."
    );
  }

  const cleanEmail =
    String(toEmail).trim();

  const cleanUserName =
    String(userName || "Farmer").trim();

  // ----------------------------------------------------------
  // Build email content
  // ----------------------------------------------------------

  const htmlContent =
    buildDailyReminderHtml({
      userName: cleanUserName,
      activities,
      upcomingReminders,
      appName,
    });

  const textContent =
    buildDailyReminderText({
      userName: cleanUserName,
      activities,
      upcomingReminders,
      appName,
    });

  const todayLabel =
    formatDateLabel(
      new Date().toISOString()
    );

  // ----------------------------------------------------------
  // Calculate today's activity count
  // ----------------------------------------------------------

  const totalToday =
    activities.filter((activity) => {
      if (!activity?.date) {
        return false;
      }

      return isSameCalendarDay(
        activity.date,
        new Date()
      );
    }).length;

  // ----------------------------------------------------------
  // EmailJS Template Parameters
  // ----------------------------------------------------------

  const templateParams = {
    // Recipient
    to_email: cleanEmail,
    email: cleanEmail,

    // User
    to_name: cleanUserName,
    user_name: cleanUserName,
    name: cleanUserName,

    // Application
    app_name: appName,

    // Email
    subject:
      `Your Daily Farm Reminders - ${todayLabel}`,

    // Content
    message: textContent,
    reminder_text: textContent,
    reminders: textContent,

    // HTML content
    html_content: htmlContent,

    // Date
    today_date: todayLabel,
    current_date: todayLabel,

    // Statistics
    total_today: totalToday,
    total_upcoming: upcomingReminders.length,
  };

  // ----------------------------------------------------------
  // Debug information
  // ----------------------------------------------------------

  console.log(
    "[EmailJS] Sending daily reminder...",
    {
      serviceId: SERVICE_ID,
      templateId: TEMPLATE_ID,
      recipient: cleanEmail,
      userName: cleanUserName,
      totalToday,
      totalUpcoming:
        upcomingReminders.length,
    }
  );

  // ----------------------------------------------------------
  // Send email
  // ----------------------------------------------------------

  try {
    const result = await emailjs.send(
      SERVICE_ID,
      TEMPLATE_ID,
      templateParams,
      {
        publicKey: PUBLIC_KEY,
      }
    );

    console.log(
      "[EmailJS] Reminder sent successfully:",
      result.status,
      result.text
    );

    return result;
  } catch (error) {
    console.error(
      "[EmailJS] Failed to send reminder:",
      error
    );

    const errorMessage =
      error?.text ||
      error?.message ||
      "EmailJS delivery failed.";

    const errorStatus =
      error?.status ||
      "unknown";

    throw new Error(
      `EmailJS error (${errorStatus}): ${errorMessage}`
    );
  }
};

// ============================================================
// Mailto Fallback
// ============================================================

export const buildMailtoLink = ({
  toEmail,
  subject,
  body,
}) => {
  const safeTo = toEmail || "";

  const safeSubject = subject
    ? encodeURIComponent(subject)
    : "";

  const safeBody = body
    ? encodeURIComponent(body)
    : "";

  let link = `mailto:${safeTo}`;

  const params = [];

  if (safeSubject) {
    params.push(
      `subject=${safeSubject}`
    );
  }

  if (safeBody) {
    params.push(
      `body=${safeBody}`
    );
  }

  if (params.length > 0) {
    link += `?${params.join("&")}`;
  }

  return link;
};

// ============================================================
// Default Export
// ============================================================

export default {
  sendDailyReminderEmail,
  buildDailyReminderHtml,
  buildDailyReminderText,
  buildMailtoLink,
};