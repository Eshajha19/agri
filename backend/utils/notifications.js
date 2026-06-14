import emailjs from "@emailjs/browser";

export function sendEmailNotification(toEmail, subject, message) {
  emailjs.send(
    import.meta.env.VITE_EMAILJS_SERVICE_ID,
    import.meta.env.VITE_EMAILJS_TEMPLATE_ID,
    {
      to_email: toEmail,
      subject,
      message
    },
    import.meta.env.VITE_EMAILJS_PUBLIC_KEY
  )
  .then(() => console.log("Email sent"))
  .catch(err => console.error("Email failed:", err));
}
