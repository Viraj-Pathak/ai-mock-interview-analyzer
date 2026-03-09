# PrepIQ 🎯
### AI-Powered Mock Interviews for STEM Careers

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20App-blue?style=for-the-badge)](https://ai-mock-interview-analyzer-vkdh.vercel.app/login)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Viraj-Pathak/ai-mock-interview-analyzer)
[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=for-the-badge&logo=vercel)](https://vercel.com)

---

## 📌 Overview

**PrepIQ** is a full-stack AI-powered mock interview platform designed specifically for STEM job seekers. It simulates realistic interview sessions, analyzes your responses, and provides personalized, actionable feedback — helping you walk into every interview with confidence.

Whether you're preparing for a software engineering role, a data science position, or any other STEM field, PrepIQ tailors the experience to your target role and experience level.

---

## ✨ Features

- 🤖 **AI-Driven Interview Questions** — Dynamically generated questions based on your chosen STEM role and experience level
- 📊 **Performance Analysis** — Detailed scoring and feedback on communication, technical knowledge, and problem-solving
- 🔐 **User Authentication** — Secure account registration and login system
- 📝 **Interview History** — Track your progress and review past sessions
- 🎯 **STEM-Focused** — Tailored question banks and evaluation criteria for technical careers
- 📱 **Responsive Design** — Seamless experience across desktop and mobile devices

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React / Next.js |
| Styling | Tailwind CSS |
| AI / LLM | Claude / Gemini API |
| Backend | Node.js / Express |
| Database | MongoDB / PostgreSQL |
| Deployment | Vercel |
| Auth | Custom JWT / Session Auth |

> **Note:** Update this table to reflect your actual stack.

---

## 🚀 Getting Started

### Prerequisites

- Node.js v18+
- npm or yarn
- API key for your AI provider (Claude / Gemini / OpenAI)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Viraj-Pathak/ai-mock-interview-analyzer.git

# 2. Navigate into the project directory
cd ai-mock-interview-analyzer

# 3. Install dependencies
npm install

# 4. Set up environment variables
cp .env.example .env.local
# Add your API keys and database URL to .env.local

# 5. Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the app.

### Environment Variables

Create a `.env.local` file in the root directory and add the following:

```env
# AI Provider
AI_API_KEY=your_api_key_here

# Database
DATABASE_URL=your_database_url_here

# Auth
JWT_SECRET=your_jwt_secret_here

# App
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

---

## 📖 Usage

1. **Register** — Create a free account at [PrepIQ](https://ai-mock-interview-analyzer-vkdh.vercel.app/login)
2. **Choose Your Role** — Select your target STEM job role and experience level
3. **Start Interview** — Answer AI-generated questions at your own pace
4. **Review Feedback** — Get detailed analysis of your performance with improvement tips
5. **Track Progress** — Revisit past interviews and measure your growth over time

---

## 📂 Project Structure

```
ai-mock-interview-analyzer/
├── app/                    # Next.js app directory
│   ├── (auth)/             # Authentication routes
│   ├── (dashboard)/        # Main app routes
│   └── api/                # API endpoints
├── components/             # Reusable UI components
├── lib/                    # Utility functions & AI logic
├── public/                 # Static assets
├── styles/                 # Global styles
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Viraj Pathak**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/viraj-pathak)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Viraj-Pathak)

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!
