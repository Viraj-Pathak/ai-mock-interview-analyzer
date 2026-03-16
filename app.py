import os
import re
import json
import datetime
from collections import Counter

from sqlalchemy import text

from flask import (
    Flask, render_template, request, redirect,
    session, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import nltk

# ── NLTK setup (works on Vercel serverless via /tmp) ─────────────────────────
NLTK_DATA = os.environ.get('NLTK_DATA', '/tmp/nltk_data')
os.makedirs(NLTK_DATA, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA)
try:
    nltk.download('vader_lexicon', download_dir=NLTK_DATA, quiet=True)
except Exception:
    pass

from analyzer import analyze_all_answers

# ── App configuration ─────────────────────────────────────────────────────────
app = Flask(__name__, instance_path='/tmp')

app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///interview.db')
# Vercel Postgres returns postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024   # 2 MB upload limit
# Reconnect on stale/suspended Neon connections (pool_pre_ping pings before use)
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,   # recycle before Neon's ~5-min idle suspension
    'connect_args': {'connect_timeout': 10},
}
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Secure cookies in production (HTTPS)
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'

db = SQLAlchemy(app)

# Custom Jinja filter so templates can do {{ json_string | from_json }}
app.jinja_env.filters['from_json'] = json.loads


# ── Models ────────────────────────────────────────────────────────────────────

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    level = db.Column(db.String(20), nullable=True)    # fresher | mid | senior
    confidence = db.Column(db.Float, nullable=False)
    relevance = db.Column(db.Float, nullable=False)
    clarity = db.Column(db.Float, nullable=True)
    cheating = db.Column(db.Float, nullable=False)
    overall = db.Column(db.Float, nullable=False)
    word_count = db.Column(db.Integer, nullable=True)
    filler_count = db.Column(db.Integer, nullable=True)
    feedback = db.Column(db.Text, nullable=True)          # JSON list of feedback strings
    per_question = db.Column(db.Text, nullable=True)      # JSON list of per-question dicts
    submitted_answers = db.Column(db.Text, nullable=True) # JSON list of raw answer strings (replay)
    multi_face_count = db.Column(db.Integer, nullable=True, default=0)
    date = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class ScheduledInterview(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(100), nullable=False)
    role         = db.Column(db.String(100), nullable=False)
    level        = db.Column(db.String(20), nullable=True)
    scheduled_at = db.Column(db.DateTime, nullable=False)
    note         = db.Column(db.Text, nullable=True)
    dismissed    = db.Column(db.Boolean, default=False)
    created_at   = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class ResumeMatch(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(100), nullable=False)
    role       = db.Column(db.String(100), nullable=False)
    score      = db.Column(db.Float, nullable=False)
    matched_kw = db.Column(db.Text, nullable=True)   # JSON list
    missing_kw = db.Column(db.Text, nullable=True)   # JSON list
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"DB create_all error: {e}")
    # Migrate new columns on existing Interview table (separate try so create_all failure
    # doesn't prevent this, and vice versa)
    try:
        with db.engine.connect() as _conn:
            _conn.execute(text("ALTER TABLE interview ADD COLUMN IF NOT EXISTS submitted_answers TEXT"))
            _conn.execute(text("ALTER TABLE interview ADD COLUMN IF NOT EXISTS multi_face_count INTEGER DEFAULT 0"))
            _conn.commit()
    except Exception as e:
        print(f"DB migration error (non-fatal): {e}")


# ── Questions bank (role → level → questions) ─────────────────────────────────

LEVELS = {
    "fresher": {"label": "Fresher",    "years": "0–2 yrs",  "icon": "🌱"},
    "mid":     {"label": "Mid-Level",  "years": "2–5 yrs",  "icon": "⚡"},
    "senior":  {"label": "Senior",     "years": "5+ yrs",   "icon": "🏆"},
}

QUESTIONS = {
    "Software Engineer": {
        "fresher": [
            "What is a REST API and what HTTP methods does it use?",
            "Explain the four pillars of Object-Oriented Programming.",
            "What is Big O notation? Give examples of O(1), O(n), and O(n²).",
            "What is a thread and how does it differ from a process?",
            "What is a database index and why is it useful?",
        ],
        "mid": [
            "Design a RESTful API for a user authentication system. What endpoints and status codes would you use?",
            "Explain the SOLID principles with a practical example for each.",
            "How would you optimize an O(n²) algorithm to O(n log n)? Give a concrete example.",
            "How do you handle thread safety in a multi-threaded application? Explain locks and race conditions.",
            "A query on a table with 10M rows is slow. Walk me through your optimization strategy.",
        ],
        "senior": [
            "You're designing a microservices architecture for an e-commerce platform. What services would you define and how do they communicate?",
            "How do you enforce good OOP and design patterns across a large engineering team? What patterns do you reach for in distributed systems?",
            "Describe how you'd handle performance degradation in a distributed system under high load.",
            "Design a concurrent job-processing queue that handles failures, retries, and priority levels.",
            "Explain your approach to database sharding. What are the tradeoffs of range-based vs. hash-based sharding?",
        ],
    },
    "Data Scientist": {
        "fresher": [
            "What is overfitting and how can you detect it?",
            "Explain the bias-variance tradeoff in your own words.",
            "What is feature engineering? Give two examples.",
            "What is k-fold cross-validation and why is it used?",
            "What is the difference between linear and logistic regression?",
        ],
        "mid": [
            "You deployed a model with 95% training accuracy but only 60% on production. What do you investigate?",
            "Compare L1 and L2 regularization. When would you use each?",
            "What techniques do you use for feature selection on a dataset with 500+ features?",
            "How does stratified k-fold differ from regular k-fold and when does it matter?",
            "Compare Ridge, Lasso, and Elastic Net. What problem does each solve?",
        ],
        "senior": [
            "Design a production ML pipeline from data ingestion to model monitoring. What are the failure points?",
            "Explain how gradient boosting works and compare XGBoost, LightGBM, and CatBoost for a tabular dataset.",
            "How would you handle automated feature engineering on a high-cardinality categorical dataset at scale?",
            "How do you apply cross-validation correctly for time-series data? What leakage pitfalls must you avoid?",
            "Describe an end-to-end approach for building and deploying a multi-output regression model in a real-time system.",
        ],
    },
    "AI/ML Engineer": {
        "fresher": [
            "What is gradient descent and what problem does it solve?",
            "Explain what a neural network is, including layers and activation functions.",
            "What are Large Language Models (LLMs) and how are they trained?",
            "What is backpropagation and why is it important?",
            "What metrics would you use to evaluate a classification model?",
        ],
        "mid": [
            "Compare Adam, SGD, and RMSprop optimizers. When would you pick each?",
            "When would you choose a CNN over an RNN and vice versa? Give use cases.",
            "Explain the difference between fine-tuning an LLM and using Retrieval-Augmented Generation (RAG).",
            "How do you address the vanishing and exploding gradient problems?",
            "Explain the precision-recall tradeoff. How do you decide where to set the classification threshold?",
        ],
        "senior": [
            "Design an end-to-end MLOps pipeline for training, serving, and monitoring a recommendation model at 100M users.",
            "Explain the transformer architecture in depth — attention mechanisms, positional encoding, and why it replaced RNNs.",
            "How would you deploy a 70B parameter LLM to serve low-latency inference requests? Discuss quantization, batching, and hardware.",
            "What techniques do you use to ensure stable training of large models? Cover learning rate schedules, gradient clipping, and mixed precision.",
            "How do you design an A/B testing framework to safely roll out a new ML model in production?",
        ],
    },
    "Product Manager": {
        "fresher": [
            "What is an MVP and why is it important?",
            "How do you prioritize a list of feature requests?",
            "Describe the four stages of the product lifecycle.",
            "What metrics would you use to measure whether a product feature is successful?",
            "How do you manage a stakeholder who disagrees with your product decision?",
        ],
        "mid": [
            "You have 10 feature requests and capacity for 3 this quarter. Walk me through how you decide.",
            "A key stakeholder wants a feature your data shows will hurt retention. How do you handle this?",
            "Your product is in the maturity stage with declining growth. What strategies do you consider?",
            "How do you write OKRs for a product team? Give an example for a SaaS product.",
            "How do you align engineering, design, and marketing around a product roadmap?",
        ],
        "senior": [
            "How do you decide whether to build, buy, or partner for a new capability?",
            "You're entering a new market with strong incumbents. How do you define a product strategy?",
            "How do you manage a portfolio of products at different lifecycle stages with limited resources?",
            "How do you drive organisational alignment when your product vision challenges the status quo?",
            "Describe a time you had to lead through ambiguity with incomplete data. How did you structure your decision?",
        ],
    },
    "Finance Analyst": {
        "fresher": [
            "What is Net Present Value (NPV) and what does a positive NPV mean?",
            "What does EBITDA measure and why do analysts use it?",
            "Explain portfolio diversification and why it reduces risk.",
            "What are the main types of financial risk and how do you identify them?",
            "What is ROI and how is it calculated?",
        ],
        "mid": [
            "When would you use IRR instead of NPV for a capital budgeting decision? What are IRR's limitations?",
            "A company reports EBITDA of $50M but has heavy capex. What adjustments do you make for a cleaner valuation?",
            "How do you apply Modern Portfolio Theory to construct an efficient portfolio for a client?",
            "Explain Value at Risk (VaR) and its limitations. How does Conditional VaR improve on it?",
            "Compare ROI and ROIC. Why might ROIC be a better measure of business quality?",
        ],
        "senior": [
            "Walk me through the three main valuation approaches in an M&A transaction and when each is most appropriate.",
            "How do you build an EBITDA bridge analysis when two companies with different accounting policies are being compared?",
            "Describe how you would construct a factor-based equity portfolio. What factors would you use and why?",
            "Design an enterprise risk management framework for a multinational company exposed to FX, credit, and liquidity risk.",
            "A PE-backed company is planning an exit. How do you evaluate the return (IRR, MOIC, DPI) and what levers improve it?",
        ],
    },
}

QUESTIONS["Frontend Developer"] = {
    "fresher": [
        "What is the difference between HTML, CSS, and JavaScript?",
        "Explain the CSS box model.",
        "What is the DOM and how does JavaScript interact with it?",
        "What is the difference between flexbox and CSS Grid?",
        "What is a JavaScript Promise?",
    ],
    "mid": [
        "Explain React hooks — useState, useEffect, and useContext with use cases.",
        "How do you optimize the performance of a React application?",
        "Compare Redux, Zustand, and React Context for state management. When do you use each?",
        "What is web accessibility (WCAG) and how do you implement it?",
        "Describe a CSS architecture strategy (BEM, CSS Modules, CSS-in-JS) and its tradeoffs.",
    ],
    "senior": [
        "How would you architect a micro-frontend system for a large e-commerce platform?",
        "Compare SSR, SSG, ISR, and CSR rendering strategies. When do you choose each in Next.js?",
        "How do you design and maintain a component library used by 10+ engineering teams?",
        "Walk me through optimizing Core Web Vitals (LCP, INP, CLS) for a content-heavy site.",
        "What are the key frontend security threats (XSS, CSRF, CSP) and how do you mitigate them?",
    ],
}
QUESTIONS["Backend Developer"] = {
    "fresher": [
        "What is the difference between REST and GraphQL?",
        "What is a database transaction and what are ACID properties?",
        "Explain the difference between authentication and authorization.",
        "What is caching and why is it used?",
        "What is the difference between SQL and NoSQL databases?",
    ],
    "mid": [
        "Design a rate limiter for an API that allows 100 requests per minute per user.",
        "How do you manage database connection pooling and why does it matter?",
        "Compare monolithic and microservices architecture — when would you choose each?",
        "Compare JWT and session-based authentication. What are the security tradeoffs?",
        "Explain the difference between horizontal and vertical scaling with examples.",
    ],
    "senior": [
        "Design a distributed task queue system that guarantees at-least-once delivery.",
        "How do you approach event-driven architecture with Kafka? Explain partitioning, consumers, and delivery guarantees.",
        "Explain CQRS and Event Sourcing — when are they appropriate and what are the operational costs?",
        "How do you design observability into a backend system (logs, metrics, traces)?",
        "Design a multi-tenant SaaS backend. How do you handle data isolation, billing, and permissions?",
    ],
}
QUESTIONS["DevOps Engineer"] = {
    "fresher": [
        "What is CI/CD and why is it important?",
        "What is Docker and what problem does containerisation solve?",
        "What is Kubernetes and what role does it play in container orchestration?",
        "What is Infrastructure as Code (IaC) and what tools are used?",
        "What is a load balancer and how does it work?",
    ],
    "mid": [
        "Design a CI/CD pipeline for a microservices application with automated testing and blue-green deployment.",
        "How do you manage secrets in a Kubernetes cluster securely?",
        "What is your approach to container resource limits, autoscaling, and health checks in Kubernetes?",
        "How do you set up monitoring and alerting for a production system? What tools and SLO/SLA concepts apply?",
        "Explain Terraform state management, remote backends, and workspace strategies for multi-environment infrastructure.",
    ],
    "senior": [
        "Design a multi-region, multi-cloud deployment strategy with automated failover for a 99.99% SLA service.",
        "How do you build an internal developer platform that reduces cognitive load for engineering teams?",
        "Describe a cloud cost optimisation strategy for a company spending $500K/month on AWS.",
        "How do you implement DevSecOps — integrating security scanning into every stage of the pipeline?",
        "What is chaos engineering and how would you implement a game-day program for a critical production system?",
    ],
}
QUESTIONS["Cybersecurity Analyst"] = {
    "fresher": [
        "Explain the CIA triad — Confidentiality, Integrity, and Availability.",
        "What are the most common types of cyberattacks? Name and explain three.",
        "What is the difference between symmetric and asymmetric encryption?",
        "What is a firewall and how does it differ from an IDS/IPS?",
        "What is vulnerability scanning and how does it differ from penetration testing?",
    ],
    "mid": [
        "Walk me through the incident response lifecycle (PICERL). How do you contain a ransomware incident?",
        "How do you design and implement a SIEM solution for a mid-size organization?",
        "Describe a penetration testing methodology for a web application. What tools do you use?",
        "How do you design a network segmentation strategy to limit lateral movement by an attacker?",
        "Explain PAM (Privileged Access Management) and how it reduces the risk of insider threats.",
    ],
    "senior": [
        "Design a zero-trust architecture for a remote-first company with 1000+ employees.",
        "How do you build a threat hunting programme from scratch? What data sources and hypotheses do you start with?",
        "How do you design and staff a Security Operations Center (SOC) for 24/7 coverage?",
        "Explain software supply chain security risks and how you mitigate them (SBOM, signing, SLSA framework).",
        "How do you build a security governance and compliance framework for a company targeting SOC 2 and ISO 27001?",
    ],
}
QUESTIONS["Data Engineer"] = {
    "fresher": [
        "What is ETL and how does it differ from ELT?",
        "What is the difference between a data warehouse and a data lake?",
        "What is Apache Spark and what type of problems does it solve?",
        "Explain the difference between batch processing and stream processing.",
        "What is a data pipeline and what are its key components?",
    ],
    "mid": [
        "Design a data ingestion pipeline to handle 1TB of daily event data with SLA requirements.",
        "How do you implement data quality checks and validation in a production pipeline?",
        "Compare star schema and snowflake schema for a data warehouse. When do you use each?",
        "How do you choose between Apache Kafka and a traditional message queue for streaming data?",
        "What is a data catalog and how does it support data governance?",
    ],
    "senior": [
        "Design a data mesh architecture for a large enterprise with 20+ domain teams.",
        "How do you architect a real-time analytics platform handling petabyte-scale event streams?",
        "How do you build a data governance framework covering lineage, quality, access control, and compliance?",
        "Design the feature engineering infrastructure for an ML platform serving 50+ data scientists.",
        "How do you approach cloud data infrastructure cost optimisation while maintaining performance SLAs?",
    ],
}
QUESTIONS["Mobile Developer"] = {
    "fresher": [
        "What is the difference between native, hybrid, and cross-platform mobile development?",
        "Explain the Android activity lifecycle or iOS UIViewController lifecycle.",
        "What is state management in React Native or Flutter and why is it important?",
        "How does push notification delivery work on iOS and Android?",
        "What is offline-first development and how do you implement local data storage?",
    ],
    "mid": [
        "How do you diagnose and fix jank (frame drops) in a mobile application?",
        "Compare Redux, MobX, and React Query for state management in React Native.",
        "What are the key mobile security best practices for storing sensitive data?",
        "How do you set up a CI/CD pipeline for mobile app releases to the App Store and Play Store?",
        "How do you implement accessibility in a mobile app for screen reader users?",
    ],
    "senior": [
        "How would you architect a large-scale React Native or Flutter app used by 10M users?",
        "How do you design a mobile SDK for third-party developers — API surface, versioning, and documentation?",
        "How do you build and scale a cross-platform performance testing and monitoring strategy?",
        "How do you design an A/B testing and feature flagging system for a mobile application?",
        "How do you manage a 50M daily active user mobile app with a 5-person engineering team?",
    ],
}
QUESTIONS["QA Engineer"] = {
    "fresher": [
        "What is the difference between manual and automated testing?",
        "Explain unit, integration, and end-to-end tests — when do you use each?",
        "What makes a good test case? Walk me through writing one.",
        "What is regression testing and why is it important?",
        "Describe the bug life cycle from discovery to closure.",
    ],
    "mid": [
        "Design a test automation framework for a REST API using a language of your choice.",
        "How do you approach performance and load testing for a web application?",
        "What is your strategy for deciding which tests to automate and which to keep manual?",
        "How do you measure test coverage and use it to prioritise testing effort?",
        "How do you triage and prioritise defects when you have 50+ open bugs before a release?",
    ],
    "senior": [
        "How do you embed quality into a CI/CD pipeline — from unit tests to production monitoring?",
        "How do you build a shift-left testing culture in an organisation that historically tested late?",
        "Describe a risk-based testing strategy for a major product release under tight deadlines.",
        "How do you design and scale a test infrastructure for 100+ microservices?",
        "How do you hire, grow, and lead a QA team across multiple product squads?",
    ],
}
QUESTIONS["Mechanical Engineer"] = {
    "fresher": [
        "Explain the difference between stress and strain, and describe Hooke's Law.",
        "What are Newton's three laws of motion and how do they apply to engineering problems?",
        "What is the first and second law of thermodynamics?",
        "How do you draw and interpret a free body diagram?",
        "What is GD&T (Geometric Dimensioning and Tolerancing) and why is it used?",
    ],
    "mid": [
        "How do you set up a Finite Element Analysis (FEA) simulation? What are the key inputs and outputs?",
        "What is Design for Manufacturability (DFM) and how does it influence your design decisions?",
        "Walk me through conducting a Failure Mode and Effects Analysis (FMEA).",
        "How do you approach thermal management in a mechanical system with high heat generation?",
        "Describe your material selection process for a structural component exposed to fatigue loading.",
    ],
    "senior": [
        "How do you apply a systems engineering approach to a complex multi-disciplinary mechanism?",
        "What is your approach to Product Lifecycle Management (PLM) and configuration control in a large programme?",
        "How do you design for reliability and maintainability in a product with a 20-year service life?",
        "How do you manage the transition from R&D prototype to high-volume manufacturing?",
        "How do you lead and align a cross-functional engineering team across mechanical, electrical, and software disciplines?",
    ],
}
QUESTIONS["Electrical Engineer"] = {
    "fresher": [
        "State Ohm's law and Kirchhoff's current and voltage laws, with an example.",
        "What is the difference between AC and DC circuits?",
        "Explain how a transistor works and its main operating regions.",
        "What is signal-to-noise ratio (SNR) and why does it matter?",
        "What is a PCB and what are its key layers?",
    ],
    "mid": [
        "How do you design a DC-DC buck converter? Explain the key component calculations.",
        "How do you perform a circuit simulation using SPICE — what can and cannot be modelled?",
        "What are EMC and EMI and how do you design a PCB to minimise electromagnetic interference?",
        "How do you choose between a microcontroller and an FPGA for a given application?",
        "How do you integrate sensors (ADC, SPI, I2C) into a hardware design and validate the interface?",
    ],
    "senior": [
        "How do you architect the electrical system for a battery-powered IoT device targeting 10-year battery life?",
        "How do you design power electronics for a high-reliability safety-critical system (medical or aerospace)?",
        "Describe hardware-software co-design — how do you partition functionality between hardware and firmware?",
        "How do you manage electrical safety, compliance (CE, UL), and EMC testing for a product entering global markets?",
        "How do you run bring-up and validation of a complex multi-board electrical system?",
    ],
}
QUESTIONS["Embedded Systems Engineer"] = {
    "fresher": [
        "What is the difference between a microcontroller and a microprocessor?",
        "What is an RTOS and why is it used in embedded systems?",
        "Explain the difference between I2C, SPI, and UART communication protocols.",
        "What is memory-mapped I/O and how do you use it to control peripherals?",
        "What is a bootloader and what is its role in an embedded system?",
    ],
    "mid": [
        "How do you design task scheduling and prioritisation in an RTOS application?",
        "Walk me through writing a peripheral driver (e.g., SPI) in bare-metal C.",
        "How do you debug an embedded system with no OS — what tools and techniques do you use?",
        "How do you manage heap and stack memory in a resource-constrained microcontroller?",
        "How do you implement and validate a CAN bus or Modbus communication stack?",
    ],
    "senior": [
        "How do you design firmware architecture for a safety-critical embedded system (IEC 61508 / ISO 26262)?",
        "How do you design a robust over-the-air (OTA) firmware update system for 1M field-deployed devices?",
        "How do you define the hardware-software interface (BSP/HAL) to maximise portability across MCU families?",
        "What techniques do you use to reduce power consumption to achieve 10-year battery life on a coin cell?",
        "How do you harden an embedded device against physical and remote security attacks?",
    ],
}
QUESTIONS["Robotics Engineer"] = {
    "fresher": [
        "What is ROS (Robot Operating System) and what problems does it solve?",
        "Explain the difference between forward kinematics and inverse kinematics.",
        "What sensors are commonly used in robotics and what does each measure?",
        "What is a PID controller and how is it tuned?",
        "What is path planning and name two common algorithms used for it.",
    ],
    "mid": [
        "Compare RRT, A*, and Dijkstra for robot motion planning. When do you choose each?",
        "How do you fuse IMU, LIDAR, and camera data for robot localisation?",
        "How do you design a control architecture for a 6-DOF robot arm performing pick-and-place?",
        "What simulation environments do you use for robotics development and how do you validate in sim?",
        "How do you implement object detection and pose estimation for a robotic grasping task?",
    ],
    "senior": [
        "How do you architect a full robotics software stack — from perception to planning to control?",
        "How do you coordinate a fleet of autonomous mobile robots (AMRs) in a warehouse environment?",
        "How do you design and certify a collaborative robot (cobot) safety system for human-robot interaction?",
        "What is the sim-to-real gap and how do you minimise it when training robot policies in simulation?",
        "How do you take a robotics prototype through to commercial deployment — hardware, software, support, and scale?",
    ],
}
QUESTIONS["Quantitative Analyst"] = {
    "fresher": [
        "What is quantitative finance and how does it differ from traditional finance?",
        "What is a derivative and give examples of options, futures, and swaps.",
        "Explain the Black-Scholes model — what does it calculate and what are its assumptions?",
        "What is Monte Carlo simulation and how is it used in finance?",
        "What is backtesting and what are the key risks when interpreting backtest results?",
    ],
    "mid": [
        "Compare delta hedging and vega hedging for an options portfolio. When do you rebalance?",
        "How do you construct a mean-variance efficient portfolio with real-world constraints (turnover, sector limits)?",
        "Explain how you build a multi-factor equity model. What factors would you include and why?",
        "How do you measure and manage VaR for a fixed-income portfolio? What stress scenarios do you use?",
        "Describe a pairs trading or statistical arbitrage strategy — how do you identify pairs and manage risk?",
    ],
    "senior": [
        "Design the technology and data infrastructure for a high-frequency trading system with sub-millisecond latency.",
        "How do you research, validate, and deploy a new alpha factor at a quantitative hedge fund?",
        "What is model risk and how do you build a model risk management framework for a trading desk?",
        "How do you apply machine learning in a systematic trading strategy while controlling for overfitting?",
        "How do you structure and lead a quant research team — hiring, process, and performance attribution?",
    ],
}
QUESTIONS["Bioinformatics Analyst"] = {
    "fresher": [
        "What is bioinformatics and how is it applied in genomics research?",
        "Explain the central dogma of molecular biology — DNA to RNA to protein.",
        "What is sequence alignment and why is it important? Name one algorithm used.",
        "What are biological databases (NCBI, UniProt, Ensembl) and what data do they contain?",
        "What is a BLAST search and how do you interpret its output?",
    ],
    "mid": [
        "Walk me through an RNA-seq analysis pipeline from raw reads to differentially expressed genes.",
        "How do you perform variant calling from whole-genome sequencing data? What tools do you use?",
        "Compare AlphaFold and homology modelling for protein structure prediction — when do you use each?",
        "How do you control for multiple testing in a genome-wide association study (GWAS)?",
        "Compare Snakemake and Nextflow for bioinformatics workflow management. What are the tradeoffs?",
    ],
    "senior": [
        "How do you integrate multi-omics data (genomics, transcriptomics, proteomics) to identify disease biomarkers?",
        "Design a cloud-based genomics data platform to store, process, and share whole-genome data for 100K patients.",
        "How do you apply machine learning for drug target identification and compound prioritisation?",
        "What regulatory and ethical considerations apply when running a clinical bioinformatics pipeline (HIPAA, GDPR, CLIA)?",
        "How do you build and lead a bioinformatics research platform team that serves wet-lab scientists at scale?",
    ],
}

ROLE_ICONS = {
    "Software Engineer":       "💻",
    "Frontend Developer":      "🖥️",
    "Backend Developer":       "⚙️",
    "Data Scientist":          "📊",
    "Data Engineer":           "🔧",
    "AI/ML Engineer":          "🤖",
    "DevOps Engineer":         "☁️",
    "Cybersecurity Analyst":   "🔐",
    "Mobile Developer":        "📱",
    "QA Engineer":             "🧪",
    "Mechanical Engineer":     "⚙️",
    "Electrical Engineer":     "⚡",
    "Embedded Systems Engineer":"🔌",
    "Robotics Engineer":       "🦾",
    "Quantitative Analyst":    "📈",
    "Bioinformatics Analyst":  "🧬",
    "Product Manager":         "🎯",
    "Finance Analyst":         "💰",
}


# ── Role slug helpers ─────────────────────────────────────────────────────────

def to_slug(role: str) -> str:
    """'AI/ML Engineer' → 'ai-ml-engineer'"""
    return re.sub(r'[^a-z0-9]+', '-', role.lower()).strip('-')

ROLE_SLUGS = {to_slug(r): r for r in QUESTIONS}  # slug → display name

# Placeholder — no hand-written ideal answers; lookups return '' gracefully
IDEAL_ANSWERS = {}


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── Anti-cheat / feature constants ───────────────────────────────────────────

CODING_ROLES = {
    "Software Engineer", "Frontend Developer", "Backend Developer",
    "Data Engineer", "Data Scientist", "AI/ML Engineer",
    "Embedded Systems Engineer", "DevOps Engineer",
}

# ── Follow-up question templates ──────────────────────────────────────────────

FOLLOWUP_TEMPLATES = {
    "low_relevance": [
        "Let's go deeper — {q} Provide a specific real-world example where you applied this.",
        "Your answer missed some key concepts. Rephrase your answer to: {q} — focus on the core technical details.",
        "If you were explaining this to a junior colleague, how would you answer: {q}",
    ],
    "low_confidence": [
        "Try answering again with more conviction: {q} Lead with your conclusion first.",
        "What is the one thing you're most certain about when it comes to: {q}",
    ],
    "too_brief": [
        "Your answer was too short. Walk through your full reasoning for: {q}",
        "Expand on your answer — add a real example or describe a trade-off for: {q}",
    ],
    "general": [
        "What would you add or change in your answer to: {q}",
        "What edge cases or failure scenarios apply to: {q}",
    ],
}

STUDY_RESOURCES = {
    "low_relevance": {
        "title": "Improve Technical Depth", "icon": "📚", "priority": "HIGH",
        "actions": [
            "Review core concepts for your target role using structured study guides.",
            "Use the STAR method (Situation → Task → Action → Result) for every behavioural answer.",
            "Open the Replay page for past interviews to compare your answers with ideal answers.",
            "Re-attempt the same role after 2–3 days of focused study.",
        ],
    },
    "low_confidence": {
        "title": "Build Answer Confidence", "icon": "🎯", "priority": "HIGH",
        "actions": [
            "Record yourself answering questions aloud and listen back critically.",
            "Lead every answer with your conclusion, then explain the reasoning behind it.",
            "Replace hedging phrases like 'I think maybe' with definitive statements.",
            "Use the Dictate feature on the interview page to practice speaking answers.",
        ],
    },
    "too_brief": {
        "title": "Expand Your Answers", "icon": "📝", "priority": "MEDIUM",
        "actions": [
            "Aim for 80–120 words minimum per answer (use the word counter on the interview page).",
            "Structure every answer: definition → how it works → real example → trade-off.",
            "Draw from personal or university projects if you lack professional experience.",
            "Ask 'So what?' after each sentence — if you can't answer, add more.",
        ],
    },
    "filler_words": {
        "title": "Reduce Filler Words", "icon": "🗣", "priority": "MEDIUM",
        "actions": [
            "Identify your top 3 filler words from past interview reports on the History page.",
            "Pause silently instead of saying 'um', 'like', or 'basically'.",
            "Slow your speaking pace — rushing causes more fillers.",
            "Practice the Dictate feature to hear yourself and self-correct.",
        ],
    },
    "low_clarity": {
        "title": "Improve Answer Clarity", "icon": "💡", "priority": "MEDIUM",
        "actions": [
            "Structure every answer: key point first → reasoning → example.",
            "Define technical terms as you use them — don't assume the interviewer knows.",
            "Write a one-sentence summary of your answer in your head before expanding.",
            "Review the Replay page — reread your answers to spot unclear sentences.",
        ],
    },
}


def calculate_streak(username):
    """Return current and longest consecutive-day interview streaks."""
    records = Interview.query.filter_by(username=username).all()
    if not records:
        return {"current_streak": 0, "longest_streak": 0}

    today = datetime.datetime.now(datetime.timezone.utc).date()
    dates = sorted(set(
        (r.date if r.date.tzinfo else r.date.replace(tzinfo=datetime.timezone.utc))
        .astimezone(datetime.timezone.utc).date()
        for r in records
    ), reverse=True)

    # Current streak: walk backwards from most-recent date
    current = 0
    if dates[0] >= today - datetime.timedelta(days=1):
        current = 1
        expected = dates[0] - datetime.timedelta(days=1)
        for d in dates[1:]:
            if d == expected:
                current += 1
                expected -= datetime.timedelta(days=1)
            elif d < expected:
                break

    # Longest streak
    dates_asc = sorted(dates)
    longest, run = 1, 1
    for i in range(1, len(dates_asc)):
        if (dates_asc[i] - dates_asc[i - 1]).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1

    return {"current_streak": current, "longest_streak": longest}


def get_weak_areas(username):
    """Aggregate feedback and score patterns across all user interviews."""
    records = Interview.query.filter_by(username=username).all()
    if not records:
        return None

    feedback_counter = Counter()
    role_scores = {}
    low_counts = {"relevance": 0, "confidence": 0, "clarity": 0}

    for r in records:
        role_scores.setdefault(r.role, []).append(r.overall)
        if r.relevance < 60:
            low_counts["relevance"] += 1
        if r.confidence < 60:
            low_counts["confidence"] += 1
        if r.clarity and r.clarity < 60:
            low_counts["clarity"] += 1
        if r.per_question:
            for qr in json.loads(r.per_question):
                if qr.get("final_score", 100) < 70:
                    for fb in qr.get("feedback", []):
                        feedback_counter[fb] += 1

    def _cat(fb):
        f = fb.lower()
        if any(w in f for w in ["relevance", "relevant", "topic", "concept", "address"]):
            return "low_relevance"
        if any(w in f for w in ["confidence", "sentiment", "filler", "uncertain"]):
            return "low_confidence"
        if any(w in f for w in ["word", "brief", "short", "expand"]):
            return "too_brief"
        if any(w in f for w in ["clarity", "clear", "structure"]):
            return "low_clarity"
        return "general"

    categorized = {}
    for fb, count in feedback_counter.most_common(15):
        cat = _cat(fb)
        categorized.setdefault(cat, []).append({"text": fb, "count": count})

    role_averages = {
        role: round(sum(scores) / len(scores), 1)
        for role, scores in role_scores.items()
    }

    return {
        "categorized_feedback": categorized,
        "role_averages": role_averages,
        "low_counts": low_counts,
        "total": len(records),
    }


def generate_tailoring_tips(role, missing_kw, matched_kw, score):
    """Return section-by-section resume tailoring suggestions based on keyword gaps."""
    if score >= 80:
        assessment = "Strong match! Minor tweaks will push you to the top of the pile."
    elif score >= 55:
        assessment = "Good foundation. Adding the missing keywords should noticeably lift your score."
    else:
        assessment = "Significant gaps found. Work through each section below to close them."

    # Spread missing keywords across sections
    skills_kw   = missing_kw[:8]
    summary_kw  = missing_kw[:3]
    exp_kw      = missing_kw[3:7] or missing_kw[:4]
    project_kw  = missing_kw[-4:] or missing_kw[:4]
    anchor_kw   = matched_kw[:2] if matched_kw else [role.split()[0].lower()]

    tips = []

    if summary_kw:
        tips.append({
            'section': 'Professional Summary',
            'icon': '📝',
            'priority': 'HIGH' if score < 55 else 'MEDIUM',
            'advice': (
                f'Open with a summary that name-drops the role and its core themes. '
                f'Weave in: {", ".join(summary_kw)}.'
            ),
            'example': (
                f'Results-driven {role} with hands-on experience in '
                f'{summary_kw[0]}{"," if len(summary_kw) > 1 else ""} '
                f'{" and ".join(summary_kw[1:3])}. Passionate about building '
                f'scalable, high-quality solutions.'
            ),
        })

    if skills_kw:
        tips.append({
            'section': 'Skills / Competencies',
            'icon': '⚙️',
            'priority': 'HIGH',
            'advice': (
                f'Add a dedicated "Technical Skills" or "Core Competencies" section. '
                f'Include at minimum: {", ".join(skills_kw)}.'
            ),
            'example': 'Technical Skills: ' + ' · '.join(skills_kw),
        })

    if exp_kw:
        tips.append({
            'section': 'Work Experience Bullets',
            'icon': '💼',
            'priority': 'HIGH',
            'advice': (
                f'Rewrite 2–3 experience bullets to include: {", ".join(exp_kw[:4])}. '
                f'Lead with a strong action verb and quantify the outcome.'
            ),
            'example': (
                f'• Designed and deployed {exp_kw[0]}-based solution, improving '
                f'{exp_kw[1] if len(exp_kw) > 1 else "system performance"} by 30% '
                f'and reducing operational overhead.'
            ),
        })

    if score < 70 and project_kw:
        tips.append({
            'section': 'Projects',
            'icon': '🚀',
            'priority': 'MEDIUM',
            'advice': (
                f'Add or expand a project that showcases: {", ".join(project_kw[:3])}. '
                f'ATS systems rank candidates with concrete project evidence higher.'
            ),
            'example': (
                f'• Built end-to-end {project_kw[0]} pipeline using '
                f'{" and ".join(anchor_kw)}, demonstrating '
                f'{project_kw[1] if len(project_kw) > 1 else "key competencies"} '
                f'in a production-grade setting.'
            ),
        })

    if matched_kw:
        tips.append({
            'section': 'Strengthen What You Already Have',
            'icon': '✅',
            'priority': 'LOW',
            'advice': (
                f'You already matched {len(matched_kw)} keywords '
                f'({", ".join(matched_kw[:4])}{"…" if len(matched_kw) > 4 else ""}). '
                f'Make sure they appear in both your summary and at least two experience bullets '
                f'so ATS parsers score them with full weight.'
            ),
            'example': '',
        })

    return {'assessment': assessment, 'tips': tips}


def generate_followups(per_question_results, questions, role, level):
    """Generate targeted follow-up questions for answers scoring below 70."""
    followups = []
    ideal = IDEAL_ANSWERS.get(role, {}).get(level, {})

    for i, qr in enumerate(per_question_results):
        if qr.get("final_score", 100) >= 70 or i >= len(questions):
            continue
        q = questions[i]
        rel = qr.get("relevance_score", 100)
        conf = qr.get("confidence_score", 100)
        wc = qr.get("word_count", 50)

        if rel < 50:
            cat = "low_relevance"
        elif conf < 50:
            cat = "low_confidence"
        elif wc < 30:
            cat = "too_brief"
        else:
            cat = "general"

        templates = FOLLOWUP_TEMPLATES[cat]
        followup_q = templates[i % len(templates)].format(q=q)

        hint = ""
        ideal_text = ideal.get(q, "")
        if ideal_text:
            hint = (ideal_text.split(".")[0] + ".") if "." in ideal_text else ideal_text[:120]
            hint = hint[:150]

        followups.append({
            "question_index": i,
            "followup_question": followup_q,
            "hint": hint,
            "score": qr.get("final_score", 0),
        })
    return followups


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html', tab='login')

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash('Invalid username or password.', 'error')
            return render_template('login.html', tab='login')

        session['username'] = username
        flash(f'Welcome back, {username}!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html', tab='login')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()

        if not username or not password or not confirm:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html', tab='register')

        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template('login.html', tab='register')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('login.html', tab='register')

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('login.html', tab='register')

        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.', 'error')
            return render_template('login.html', tab='register')

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session['username'] = username
        flash(f'Account created! Welcome, {username}!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html', tab='register')


@app.route('/logout')
def logout():
    username = session.get('username', '')
    session.clear()
    flash(f'Goodbye, {username}! See you soon.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    username = session['username']
    interviews = Interview.query.filter_by(username=username).all()

    stats = {}
    if interviews:
        scores = [i.overall for i in interviews]
        stats = {
            'total': len(interviews),
            'best': round(max(scores), 1),
            'avg': round(sum(scores) / len(scores), 1),
            'last_role': interviews[-1].role,
        }

    try:
        streak = calculate_streak(username)
    except Exception:
        streak = None

    # Use naive UTC so arithmetic with PostgreSQL's naive datetimes works in templates
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    try:
        upcoming = ScheduledInterview.query.filter_by(
            username=username, dismissed=False
        ).filter(
            ScheduledInterview.scheduled_at <= now + datetime.timedelta(hours=48),
            ScheduledInterview.scheduled_at >= now,
        ).order_by(ScheduledInterview.scheduled_at).all()
    except Exception:
        upcoming = []

    return render_template(
        'dashboard.html',
        roles=QUESTIONS.keys(),
        icons=ROLE_ICONS,
        levels=LEVELS,
        stats=stats,
        streak=streak,
        upcoming=upcoming,
        now=now,
        role_slugs={r: to_slug(r) for r in QUESTIONS},
        coding_roles=CODING_ROLES,
    )


@app.route('/interview/<role_slug>/<level>', methods=['GET', 'POST'])
@login_required
def interview(role_slug, level):
    role = ROLE_SLUGS.get(role_slug)
    if role is None or level not in LEVELS:
        flash('Invalid role or experience level.', 'error')
        return redirect(url_for('dashboard'))

    questions = QUESTIONS[role][level]
    level_info = LEVELS[level]

    if request.method == 'POST':
        answers = request.form.getlist('answers')
        camera = request.form.get('camera', 'off')
        try:
            tab_switches = int(request.form.get('tab_switches', 0))
        except (ValueError, TypeError):
            tab_switches = 0

        # Clamp tab switches to reasonable range
        tab_switches = min(tab_switches, 20)

        # Parse paste detection data
        paste_counts_str = request.form.get('paste_counts', '')
        try:
            paste_counts = [int(x) for x in paste_counts_str.split(',') if x.strip()]
        except (ValueError, TypeError):
            paste_counts = []
        pastes_detected = sum(1 for c in paste_counts if c > 0)

        # Parse gaze-tracking data (collected by MediaPipe in browser)
        try:
            gaze_look_away = min(int(request.form.get('gaze_look_away', 0)), 20)
            gaze_no_face   = min(int(request.form.get('gaze_no_face', 0)), 120)
        except (ValueError, TypeError):
            gaze_look_away, gaze_no_face = 0, 0
        gaze_penalty = gaze_look_away * 5 + (gaze_no_face // 5) * 2

        # Multiple-face detection penalty
        try:
            multi_face = min(int(request.form.get('multi_face_count', 0)), 20)
        except (ValueError, TypeError):
            multi_face = 0

        # Background audio alerts penalty
        try:
            audio_alerts = min(int(request.form.get('audio_alerts', 0)), 10)
        except (ValueError, TypeError):
            audio_alerts = 0

        if len(answers) != len(questions):
            flash('Please answer all questions.', 'error')
            return render_template('interview.html', role=role, level=level,
                                   level_info=level_info, questions=questions,
                                   show_editor=(role in CODING_ROLES))

        # Deep NLP analysis via analyzer.py
        analysis = analyze_all_answers(answers, role, level, questions)

        confidence = analysis['avg_confidence']
        relevance = analysis['avg_relevance']
        clarity = analysis['avg_clarity']
        filler_count = analysis['total_filler_count']
        word_count = analysis['total_word_count']

        # Apply cheating penalties
        cheating_penalty = tab_switches * 5
        cheating_penalty += pastes_detected * 10
        cheating_penalty += gaze_penalty
        cheating_penalty += multi_face * 8
        cheating_penalty += audio_alerts * 3
        if camera != 'on':
            cheating_penalty += 10

        confidence = max(0, confidence - cheating_penalty)
        overall = round(
            max(0, min(100, (relevance * 0.45) + (confidence * 0.35) + (clarity * 0.20))), 2
        )

        # Aggregate feedback across all questions
        all_feedback = []
        for q_result in analysis['per_question']:
            for fb in q_result.get('feedback', []):
                if fb not in all_feedback:
                    all_feedback.append(fb)

        base_fields = dict(
            username=session['username'],
            role=role,
            level=level,
            confidence=round(confidence, 2),
            relevance=round(relevance, 2),
            clarity=round(clarity, 2),
            cheating=round(cheating_penalty, 2),
            overall=overall,
            word_count=word_count,
            filler_count=filler_count,
            feedback=json.dumps(all_feedback),
            per_question=json.dumps(analysis['per_question']),
        )
        try:
            record = Interview(
                **base_fields,
                submitted_answers=json.dumps(answers),
                multi_face_count=multi_face,
            )
            db.session.add(record)
            db.session.commit()
        except Exception:
            db.session.rollback()
            # New columns may not exist yet — save without them
            record = Interview(**base_fields)
            db.session.add(record)
            db.session.commit()

        return redirect(url_for('result', interview_id=record.id))

    return render_template('interview.html', role=role, level=level,
                           level_info=level_info, questions=questions,
                           show_editor=(role in CODING_ROLES))


@app.route('/result/<int:interview_id>')
@login_required
def result(interview_id):
    record = Interview.query.filter_by(
        id=interview_id, username=session['username']
    ).first_or_404()

    feedback = json.loads(record.feedback) if record.feedback else []
    per_question = json.loads(record.per_question) if record.per_question else []
    rec_level = record.level or 'fresher'
    questions = QUESTIONS.get(record.role, {}).get(rec_level, [])
    level_info = LEVELS.get(rec_level, LEVELS['fresher'])

    followups = generate_followups(per_question, questions, record.role, rec_level)

    return render_template(
        'result.html',
        record=record,
        feedback=feedback,
        per_question=per_question,
        questions=questions,
        level_info=level_info,
        followups=followups,
    )


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    records = (
        Interview.query
        .filter_by(username=session['username'])
        .order_by(Interview.date.desc())
        .paginate(page=page, per_page=8, error_out=False)
    )
    return render_template('history.html', records=records)


@app.route('/history/delete/<int:interview_id>', methods=['POST'])
@login_required
def delete_interview(interview_id):
    record = Interview.query.filter_by(
        id=interview_id, username=session['username']
    ).first_or_404()
    db.session.delete(record)
    db.session.commit()
    flash('Interview record deleted.', 'info')
    return redirect(url_for('history'))


# ── Progress tracker ──────────────────────────────────────────────────────────

@app.route('/progress/<role_slug>')
@login_required
def progress(role_slug):
    role = ROLE_SLUGS.get(role_slug)
    if not role:
        flash('Unknown role.', 'error')
        return redirect(url_for('dashboard'))

    records = Interview.query.filter_by(
        username=session['username'], role=role
    ).order_by(Interview.date.asc()).all()

    chart_data = {
        'dates':      [r.date.strftime('%b %d') for r in records],
        'overall':    [r.overall    for r in records],
        'relevance':  [r.relevance  for r in records],
        'confidence': [r.confidence for r in records],
        'clarity':    [r.clarity or 0 for r in records],
    }
    all_roles = [(to_slug(r), r) for r in QUESTIONS.keys()]
    return render_template('progress.html', role=role, role_slug=role_slug,
                           records=records, chart_data=json.dumps(chart_data),
                           all_roles=all_roles)


# ── Weak areas ────────────────────────────────────────────────────────────────

@app.route('/weak-areas')
@login_required
def weak_areas():
    data = get_weak_areas(session['username'])
    return render_template('weak_areas.html', data=data, study_resources=STUDY_RESOURCES)


# ── Study plan ────────────────────────────────────────────────────────────────

@app.route('/study-plan')
@login_required
def study_plan():
    data = get_weak_areas(session['username'])
    if not data:
        return render_template('study_plan.html', plan=[], total=0)

    # Build ordered plan: present categories with most issues first
    order = ['low_relevance', 'low_confidence', 'too_brief', 'filler_words', 'low_clarity']
    cat_counts = {cat: len(items) for cat, items in data['categorized_feedback'].items()}
    # also factor in low_counts scores
    if data['low_counts']['relevance'] > data['total'] * 0.4:
        cat_counts['low_relevance'] = cat_counts.get('low_relevance', 0) + 5
    if data['low_counts']['confidence'] > data['total'] * 0.4:
        cat_counts['low_confidence'] = cat_counts.get('low_confidence', 0) + 5

    plan = []
    seen = set()
    for cat in sorted(cat_counts, key=lambda c: -cat_counts.get(c, 0)):
        if cat in STUDY_RESOURCES and cat not in seen:
            plan.append(STUDY_RESOURCES[cat])
            seen.add(cat)
    # Always include at least general confidence tip
    for cat in order:
        if cat not in seen and cat in STUDY_RESOURCES:
            plan.append(STUDY_RESOURCES[cat])
            seen.add(cat)

    return render_template('study_plan.html', plan=plan, total=data['total'],
                           role_averages=data['role_averages'])


# ── Interview replay ───────────────────────────────────────────────────────────

@app.route('/replay/<int:interview_id>')
@login_required
def replay(interview_id):
    record = Interview.query.filter_by(
        id=interview_id, username=session['username']
    ).first_or_404()

    rec_level = record.level or 'fresher'
    questions = QUESTIONS.get(record.role, {}).get(rec_level, [])
    per_question = json.loads(record.per_question) if record.per_question else []
    submitted = json.loads(record.submitted_answers) if record.submitted_answers else []
    ideal = IDEAL_ANSWERS.get(record.role, {}).get(rec_level, {})
    level_info = LEVELS.get(rec_level, LEVELS['fresher'])

    qa_pairs = []
    for i, q in enumerate(questions):
        qa_pairs.append({
            'question':     q,
            'your_answer':  submitted[i] if i < len(submitted) else '',
            'ideal_answer': ideal.get(q, ''),
            'scores':       per_question[i] if i < len(per_question) else {},
        })

    return render_template('replay.html', record=record, qa_pairs=qa_pairs,
                           level_info=level_info)


# ── Resume match ───────────────────────────────────────────────────────────────

@app.route('/resume-match', methods=['GET', 'POST'])
@login_required
def resume_match():
    try:
        past = ResumeMatch.query.filter_by(
            username=session['username']
        ).order_by(ResumeMatch.created_at.desc()).limit(10).all()
    except Exception:
        past = []

    if request.method == 'POST':
        role_slug = request.form.get('role_slug', '')
        role = ROLE_SLUGS.get(role_slug)
        uploaded = request.files.get('resume')

        if not role or not uploaded:
            flash('Please select a role and upload your resume.', 'error')
            return render_template('resume_match.html', past=past,
                                   roles=[(to_slug(r), r) for r in QUESTIONS.keys()])

        # Extract text
        resume_text = ''
        fname = uploaded.filename.lower()
        try:
            if fname.endswith('.pdf'):
                try:
                    import pypdf
                    import io as _io
                    raw = uploaded.read()
                    reader = pypdf.PdfReader(_io.BytesIO(raw))
                    resume_text = ' '.join(
                        page.extract_text() or '' for page in reader.pages
                    )
                except ImportError:
                    flash('PDF support not available. Please upload a .txt file.', 'error')
                    return render_template('resume_match.html', past=past,
                                           roles=[(to_slug(r), r) for r in QUESTIONS.keys()])
            else:
                resume_text = uploaded.read().decode('utf-8', errors='ignore')
        except Exception:
            flash('Could not parse the uploaded file. Please use PDF or TXT.', 'error')
            return render_template('resume_match.html', past=past,
                                   roles=[(to_slug(r), r) for r in QUESTIONS.keys()])

        if len(resume_text.strip()) < 50:
            flash('Resume text is too short or could not be extracted.', 'error')
            return render_template('resume_match.html', past=past,
                                   roles=[(to_slug(r), r) for r in QUESTIONS.keys()])

        # Build role corpus from the role's interview questions
        corpus_parts = []
        for lvl_q_list in QUESTIONS.get(role, {}).values():
            corpus_parts.extend(lvl_q_list)
        role_corpus = ' '.join(corpus_parts)

        # TF-IDF cosine similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vect = TfidfVectorizer(stop_words='english', max_features=500)
        try:
            tfidf = vect.fit_transform([role_corpus, resume_text])
            score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100
        except Exception:
            score = 0.0

        # Keyword gap analysis
        try:
            feature_names = np.array(vect.get_feature_names_out())
            role_weights = tfidf[0].toarray()[0]
            top_idx = role_weights.argsort()[-40:][::-1]
            top_kw = set(feature_names[top_idx])
            resume_tokens = set(re.findall(r'[a-z]{3,}', resume_text.lower()))
            matched = sorted(top_kw & resume_tokens)
            missing = sorted(top_kw - resume_tokens)[:20]
        except Exception:
            matched, missing = [], []

        rm = ResumeMatch(
            username=session['username'],
            role=role,
            score=round(score, 1),
            matched_kw=json.dumps(matched),
            missing_kw=json.dumps(missing),
        )
        try:
            db.session.add(rm)
            db.session.commit()
        except Exception:
            db.session.rollback()

        try:
            past = ResumeMatch.query.filter_by(
                username=session['username']
            ).order_by(ResumeMatch.created_at.desc()).limit(10).all()
        except Exception:
            past = []
        tailoring = generate_tailoring_tips(role, missing, matched, score)
        flash(f'Resume matched against {role} — score: {round(score, 1)}%', 'success')
        return render_template('resume_match.html', past=past,
                               roles=[(to_slug(r), r) for r in QUESTIONS.keys()],
                               tailoring=tailoring, match_role=role,
                               match_score=round(score, 1),
                               matched=matched, missing=missing)

    return render_template('resume_match.html', past=past,
                           roles=[(to_slug(r), r) for r in QUESTIONS.keys()])


# ── Schedule ───────────────────────────────────────────────────────────────────

@app.route('/schedule', methods=['GET', 'POST'])
@login_required
def schedule():
    username = session['username']
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    if request.method == 'POST':
        action = request.form.get('action', 'create')

        if action == 'dismiss':
            sid = request.form.get('schedule_id', type=int)
            rec = ScheduledInterview.query.filter_by(id=sid, username=username).first()
            if rec:
                rec.dismissed = True
                db.session.commit()
            return redirect(url_for('schedule'))

        role_slug = request.form.get('role_slug', '')
        role = ROLE_SLUGS.get(role_slug)
        level = request.form.get('level', 'fresher')
        dt_str = request.form.get('scheduled_at', '')
        note = request.form.get('note', '').strip()

        if not role or level not in LEVELS or not dt_str:
            flash('Please fill all required fields.', 'error')
        else:
            try:
                scheduled_at = datetime.datetime.fromisoformat(dt_str).replace(tzinfo=None)
                max_future = now + datetime.timedelta(days=365)
                if scheduled_at <= now:
                    flash('Scheduled time must be in the future.', 'error')
                elif scheduled_at > max_future:
                    flash('Cannot schedule more than 1 year in advance.', 'error')
                else:
                    db.session.add(ScheduledInterview(
                        username=username, role=role, level=level,
                        scheduled_at=scheduled_at, note=note or None,
                    ))
                    db.session.commit()
                    flash('Interview scheduled!', 'success')
            except ValueError:
                flash('Invalid date/time format.', 'error')
        return redirect(url_for('schedule'))

    try:
        upcoming = ScheduledInterview.query.filter_by(
            username=username, dismissed=False
        ).filter(ScheduledInterview.scheduled_at >= now).order_by(
            ScheduledInterview.scheduled_at
        ).all()
    except Exception:
        upcoming = []

    try:
        past_sched = ScheduledInterview.query.filter_by(username=username).filter(
            ScheduledInterview.scheduled_at < now
        ).order_by(ScheduledInterview.scheduled_at.desc()).limit(5).all()
    except Exception:
        past_sched = []

    return render_template('schedule.html', upcoming=upcoming, past_sched=past_sched,
                           roles=[(to_slug(r), r) for r in QUESTIONS.keys()],
                           levels=LEVELS, now=now)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template('500.html'), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    debug = os.environ.get('FLASK_ENV', 'development') != 'production'
    app.run(debug=debug)
