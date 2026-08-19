import emailjs from "@emailjs/browser";

const PUBLIC_KEY = import.meta.env.VITE_EMAILJS_PUBLIC_KEY;
const SERVICE_ID = import.meta.env.VITE_EMAILJS_SERVICE_ID;
const TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID;

if (PUBLIC_KEY) {
  emailjs.init(PUBLIC_KEY);
}

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

const escapeHtml = (value) => {
  if (!value) return "";
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
};

const formatDateLabel = (dateStr) => {
  try {
    return new Date(dateStr).toLocaleDateString("en-IN", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
};

const formatTimeLabel = (timeStr) => {
  if (!timeStr) return "";
  try {
    const [hours, minutes] = timeStr.split(":").map(Number);
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

export const buildDailyReminderHtml = ({
  userName = "Farmer",
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  const todayLabel = formatDateLabel(new Date().toISOString());
  const todayActivities = activities.filter((a) => {
    if (!a.date) return false;
    const activityDate = new Date(a.date);
    const today = new Date();
    return (
      activityDate.getDate() === today.getDate() &&
      activityDate.getMonth() === today.getMonth() &&
      activityDate.getFullYear() === today.getFullYear()
    );
  });

  const sortedToday = [...todayActivities].sort((a, b) => {
    const aTime = a.time || "00:00";
    const bTime = b.time || "00:00";
    return aTime.localeCompare(bTime);
  });

  const upcoming = [...upcomingReminders]
    .filter((a) => {
      if (!a.date) return false;
      const activityDate = new Date(a.date);
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      return activityDate > today;
    })
    .sort((a, b) => {
      const aDate = `${a.date || ""}T${a.time || "00:00"}`;
      const bDate = `${b.date || ""}T${b.time || "00:00"}`;
      return aDate.localeCompare(bDate);
    })
    .slice(0, 5);

  const renderActivityRow = (activity) => {
    const typeLabel = ACTIVITY_TYPE_LABELS[activity.type] || "Other";
    const color = ACTIVITY_TYPE_COLORS[activity.type] || "#6b7280";
    const timeLabel = formatTimeLabel(activity.time);
    const description = escapeHtml(activity.description || "");

    return `
      <tr>
        <td style="padding: 12px 16px; border-bottom: 1px solid #f0fdf4; vertical-align: top;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <span style="
              display: inline-flex;
              align-items: center;
              gap: 6px;
              background: ${color}18;
              color: ${color};
              padding: 4px 10px;
              border-radius: 20px;
              font-size: 12px;
              font-weight: 600;
              white-space: nowrap;
            ">
              ${typeLabel}
            </span>
            ${timeLabel ? `<span style="color: #6b7280; font-size: 13px;">${timeLabel}</span>` : ""}
          </div>
          <div style="margin-top: 6px; font-weight: 600; color: #111827; font-size: 15px;">
            ${escapeHtml(activity.title)}
          </div>
          ${description ? `<div style="margin-top: 4px; color: #4b5563; font-size: 13px; line-height: 1.5;">${description}</div>` : ""}
          ${activity.cropType ? `<div style="margin-top: 4px; color: #059669; font-size: 12px; font-weight: 500;">Crop: ${escapeHtml(activity.cropType)}</div>` : ""}
        </td>
      </tr>
    `;
  };

  const renderUpcomingRow = (activity) => {
    const typeLabel = ACTIVITY_TYPE_LABELS[activity.type] || "Other";
    const color = ACTIVITY_TYPE_COLORS[activity.type] || "#6b7280";
    const dateLabel = formatDateLabel(activity.date);
    const timeLabel = formatTimeLabel(activity.time);

    return `
      <tr>
        <td style="padding: 10px 16px; border-bottom: 1px solid #f3f4f6; vertical-align: top;">
          <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span style="
              display: inline-flex;
              align-items: center;
              background: ${color}18;
              color: ${color};
              padding: 2px 8px;
              border-radius: 12px;
              font-size: 11px;
              font-weight: 600;
            ">
              ${typeLabel}
            </span>
            <span style="font-weight: 500; color: #111827; font-size: 14px;">
              ${escapeHtml(activity.title)}
            </span>
            <span style="color: #6b7280; font-size: 12px;">
              ${dateLabel}${timeLabel ? ` at ${timeLabel}` : ""}
            </span>
          </div>
        </td>
      </tr>
    `;
  };

  const todayRows = sortedToday.length > 0
    ? sortedToday.map(renderActivityRow).join("")
    : `<tr><td style="padding: 20px 16px; text-align: center; color: #9ca3af; font-size: 14px;">No activities scheduled for today.</td></tr>`;

  const upcomingRows = upcoming.length > 0
    ? upcoming.map(renderUpcomingRow).join("")
    : `<tr><td style="padding: 16px; text-align: center; color: #9ca3af; font-size: 14px;">No upcoming reminders in the next few days.</td></tr>`;

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Daily Farm Reminders</title>
    </head>
    <body style="margin: 0; padding: 0; background-color: #f9fafb; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f9fafb; padding: 24px 0;">
        <tr>
          <td align="center">
            <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); max-width: 600px; width: 100%;">
              <tr>
                <td style="background: linear-gradient(135deg, #059669 0%, #10b981 100%); padding: 28px 32px; text-align: center;">
                  <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 700; letter-spacing: 0.3px;">
                    ${escapeHtml(appName)}
                  </h1>
                  <p style="margin: 8px 0 0; color: #d1fae5; font-size: 14px; font-weight: 500;">
                    Daily Farm Reminder
                  </p>
                </td>
              </tr>
              <tr>
                <td style="padding: 28px 32px 12px;">
                  <p style="margin: 0; color: #111827; font-size: 16px; font-weight: 600;">
                    Hello, ${escapeHtml(userName)}!
                  </p>
                  <p style="margin: 6px 0 0; color: #4b5563; font-size: 14px; line-height: 1.6;">
                    Here is your farm schedule for <strong>${todayLabel}</strong>. Stay on top of your activities and keep your crops thriving.
                  </p>
                </td>
              </tr>
              <tr>
                <td style="padding: 0 32px;">
                  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f9fafb; border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb;">
                    <tr>
                      <td style="padding: 14px 16px; background-color: #ecfdf5; border-bottom: 1px solid #d1fae5;">
                        <span style="color: #065f46; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                          Today's Activities
                        </span>
                      </td>
                    </tr>
                    ${todayRows}
                  </table>
                </td>
              </tr>
              <tr>
                <td style="padding: 20px 32px 0;">
                  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e5e7eb;">
                    <tr>
                      <td style="padding: 14px 16px; background-color: #eff6ff; border-bottom: 1px solid #dbeafe;">
                        <span style="color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                          Upcoming Reminders
                        </span>
                      </td>
                    </tr>
                    ${upcomingRows}
                  </table>
                </td>
              </tr>
              <tr>
                <td style="padding: 24px 32px 28px; text-align: center;">
                  <p style="margin: 0; color: #9ca3af; font-size: 12px; line-height: 1.6;">
                    You are receiving this email because you have active reminders in ${escapeHtml(appName)}.
                  </p>
                  <p style="margin: 8px 0 0; color: #9ca3af; font-size: 12px;">
                    &copy; ${new Date().getFullYear()} ${escapeHtml(appName)}. All rights reserved.
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

export const buildDailyReminderText = ({
  userName = "Farmer",
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  const today = new Date();
  const todayLabel = formatDateLabel(today.toISOString());
  const todayActivities = activities.filter((a) => {
    if (!a.date) return false;
    const activityDate = new Date(a.date);
    return (
      activityDate.getDate() === today.getDate() &&
      activityDate.getMonth() === today.getMonth() &&
      activityDate.getFullYear() === today.getFullYear()
    );
  });

  const sortedToday = [...todayActivities].sort((a, b) => {
    const aTime = a.time || "00:00";
    const bTime = b.time || "00:00";
    return aTime.localeCompare(bTime);
  });

  const upcoming = [...upcomingReminders]
    .filter((a) => {
      if (!a.date) return false;
      const activityDate = new Date(a.date);
      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);
      return activityDate > todayStart;
    })
    .sort((a, b) => {
      const aDate = `${a.date || ""}T${a.time || "00:00"}`;
      const bDate = `${b.date || ""}T${b.time || "00:00"}`;
      return aDate.localeCompare(bDate);
    })
    .slice(0, 5);

  const lines = [
    `Hello ${userName}!`,
    ``,
    `Here is your farm schedule for ${todayLabel}:`,
    ``,
    `TODAY'S ACTIVITIES`,
    `------------------`,
  ];

  if (sortedToday.length === 0) {
    lines.push(`No activities scheduled for today.`);
  } else {
    sortedToday.forEach((activity) => {
      const typeLabel = ACTIVITY_TYPE_LABELS[activity.type] || "Other";
      const timeLabel = formatTimeLabel(activity.time);
      lines.push(`${timeLabel ? `[${timeLabel}] ` : ""}${typeLabel}: ${activity.title}`);
      if (activity.description) {
        lines.push(`   ${activity.description}`);
      }
      if (activity.cropType) {
        lines.push(`   Crop: ${activity.cropType}`);
      }
      lines.push(``);
    });
  }

  lines.push(`UPCOMING REMINDERS`);
  lines.push(`------------------`);

  if (upcoming.length === 0) {
    lines.push(`No upcoming reminders in the next few days.`);
  } else {
    upcoming.forEach((activity) => {
      const typeLabel = ACTIVITY_TYPE_LABELS[activity.type] || "Other";
      const dateLabel = formatDateLabel(activity.date);
      const timeLabel = formatTimeLabel(activity.time);
      lines.push(`${typeLabel}: ${activity.title}`);
      lines.push(`   ${dateLabel}${timeLabel ? ` at ${timeLabel}` : ""}`);
      lines.push(``);
    });
  }

  lines.push(`---`);
  lines.push(`You are receiving this email because you have active reminders in ${appName}.`);
  lines.push(`© ${today.getFullYear()} ${appName}. All rights reserved.`);

  return lines.join("\n");
};

export const sendDailyReminderEmail = async ({
  toEmail,
  userName,
  activities = [],
  upcomingReminders = [],
  appName = "Fasal Saathi",
}) => {
  if (!SERVICE_ID || !TEMPLATE_ID || !PUBLIC_KEY) {
    const missing = [];
    if (!SERVICE_ID) missing.push("VITE_EMAILJS_SERVICE_ID");
    if (!TEMPLATE_ID) missing.push("VITE_EMAILJS_TEMPLATE_ID");
    if (!PUBLIC_KEY) missing.push("VITE_EMAILJS_PUBLIC_KEY");
    throw new Error(
      `EmailJS is not configured. Missing env vars: ${missing.join(", ")}. ` +
      "Add them in Vercel Dashboard → Project Settings → Environment Variables."
    );
  }

  if (!toEmail) {
    throw new Error("Recipient email is required.");
  }

  const htmlContent = buildDailyReminderHtml({
    userName: userName || "Farmer",
    activities,
    upcomingReminders,
    appName,
  });

  const textContent = buildDailyReminderText({
    userName: userName || "Farmer",
    activities,
    upcomingReminders,
    appName,
  });

  const todayLabel = formatDateLabel(new Date().toISOString());

  const templateParams = {
    to_email: toEmail,
    subject: `Your Daily Farm Reminders - ${todayLabel}`,
    message: textContent,
    html_content: htmlContent,
    user_name: userName || "Farmer",
    app_name: appName,
    today_date: todayLabel,
    total_today: activities.filter((a) => {
      if (!a.date) return false;
      const d = new Date(a.date);
      const t = new Date();
      return d.getDate() === t.getDate() && d.getMonth() === t.getMonth() && d.getFullYear() === t.getFullYear();
    }).length,
  };

  try {
    const result = await emailjs.send(SERVICE_ID, TEMPLATE_ID, templateParams);
    console.log("[EmailJS] Reminder sent successfully:", result.status, result.text);
    return result;
  } catch (error) {
    console.error("[EmailJS] Failed to send reminder:", error);
    const message = error?.text || error?.message || "EmailJS delivery failed.";
    throw new Error(`EmailJS error (${error?.status || "network"}): ${message}`);
  }
};

export const buildMailtoLink = ({ toEmail, subject, body }) => {
  const safeTo = toEmail || "";
  const safeSubject = subject ? encodeURIComponent(subject) : "";
  const safeBody = body ? encodeURIComponent(body) : "";

  let link = `mailto:${safeTo}`;
  const params = [];
  if (safeSubject) params.push(`subject=${safeSubject}`);
  if (safeBody) params.push(`body=${safeBody}`);
  if (params.length) link += `?${params.join("&")}`;

  return link;
};

export default {
  sendDailyReminderEmail,
  buildDailyReminderHtml,
  buildDailyReminderText,
  buildMailtoLink,
};
