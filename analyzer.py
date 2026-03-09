"""
NLP-based interview answer analyzer.
Ideal answers organized by role → level → question for difficulty-aware scoring.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually",
                "literally", "sort of", "kind of"]

# ── Ideal answers: role → level → question ────────────────────────────────────

IDEAL_ANSWERS = {
    "Software Engineer": {
        "fresher": {
            "What is a REST API and what HTTP methods does it use?": (
                "REST API uses HTTP methods GET POST PUT DELETE PATCH to interact with resources over the web. "
                "It is stateless meaning each request contains all needed information. "
                "Resources are identified by URLs and data is typically exchanged as JSON."
            ),
            "Explain the four pillars of Object-Oriented Programming.": (
                "The four pillars are encapsulation hiding internal state behind methods, "
                "inheritance sharing behaviour between parent and child classes, "
                "polymorphism allowing objects of different types to be used interchangeably, "
                "and abstraction hiding complexity behind interfaces."
            ),
            "What is Big O notation? Give examples of O(1), O(n), and O(n²).": (
                "Big O notation describes how an algorithm scales with input size. "
                "O(1) is constant time like accessing an array index. "
                "O(n) is linear time like looping through an array. "
                "O(n squared) is quadratic like nested loops in bubble sort."
            ),
            "What is a thread and how does it differ from a process?": (
                "A thread is the smallest unit of execution within a process. "
                "Threads share the same memory space within a process while processes have separate memory. "
                "Threads are lightweight and faster to create. Multiple threads enable concurrency within one process."
            ),
            "What is a database index and why is it useful?": (
                "A database index is a data structure that speeds up query lookups by avoiding full table scans. "
                "B-tree indexes are common for range queries and hash indexes for equality checks. "
                "Indexes improve read performance but add overhead to writes and storage."
            ),
        },
        "mid": {
            "Design a RESTful API for a user authentication system. What endpoints and status codes would you use?": (
                "POST /auth/register returns 201, POST /auth/login returns 200 with JWT token, "
                "POST /auth/logout returns 204, GET /auth/me returns 200 with user profile, "
                "POST /auth/refresh returns 200 with new token. "
                "Use 401 for unauthorized, 403 for forbidden, 400 for validation errors. "
                "Store passwords hashed with bcrypt, use HTTPS, implement rate limiting on login."
            ),
            "Explain the SOLID principles with a practical example for each.": (
                "Single Responsibility each class has one reason to change. "
                "Open Closed open for extension closed for modification use inheritance or strategy pattern. "
                "Liskov Substitution subclasses should be substitutable for base classes. "
                "Interface Segregation prefer small specific interfaces over large ones. "
                "Dependency Inversion depend on abstractions not concretions use dependency injection."
            ),
            "How would you optimize an O(n²) algorithm to O(n log n)? Give a concrete example.": (
                "Replace nested loops with sorting plus single pass or use hash maps. "
                "Example: find duplicate elements changes from O(n squared) with nested loops "
                "to O(n) with a hash set. Sorting two arrays before merging reduces comparison cost. "
                "Use divide and conquer algorithms like merge sort or binary search trees."
            ),
            "How do you handle thread safety in a multi-threaded application? Explain locks and race conditions.": (
                "Thread safety prevents race conditions where multiple threads access shared data concurrently producing inconsistent results. "
                "Use mutex locks or synchronized blocks to ensure only one thread modifies data at a time. "
                "Use atomic operations for simple counters. Prefer immutable data structures. "
                "Use thread-safe collections. Deadlocks occur when threads wait for each other — avoid by consistent lock ordering."
            ),
            "A query on a table with 10M rows is slow. Walk me through your optimization strategy.": (
                "First run EXPLAIN ANALYZE to find the bottleneck. Check if the WHERE clause columns are indexed. "
                "Add composite indexes for multi-column filters. Avoid SELECT star, fetch only needed columns. "
                "Check for N+1 query problems. Partition large tables by date or ID range. "
                "Consider caching frequently read data in Redis. Review query joins for unnecessary full scans."
            ),
        },
        "senior": {
            "You're designing a microservices architecture for an e-commerce platform. What services would you define and how do they communicate?": (
                "Define services: user-service, product-catalog, inventory, order-service, payment, notification, search. "
                "Synchronous communication via REST or gRPC for real-time requests like checkout. "
                "Asynchronous via message queues Kafka or RabbitMQ for events like order-placed or inventory-updated. "
                "Use API gateway for routing auth and rate limiting. Implement circuit breakers for resilience. "
                "Each service owns its database to avoid coupling. Use distributed tracing for observability."
            ),
            "How do you enforce good OOP and design patterns across a large engineering team? What patterns do you reach for in distributed systems?": (
                "Enforce through code reviews, linting rules, shared libraries with documented interfaces, and architectural decision records. "
                "In distributed systems use Strategy for interchangeable algorithms, Observer for event-driven communication, "
                "Saga pattern for distributed transactions, Circuit Breaker for fault tolerance, "
                "Repository pattern for data access abstraction, and CQRS for read-write separation at scale."
            ),
            "Describe how you'd handle performance degradation in a distributed system under high load.": (
                "Add horizontal scaling and load balancing. Implement caching at multiple layers CDN application database. "
                "Use read replicas for database read scaling. Apply rate limiting and back-pressure. "
                "Profile with distributed tracing to find bottlenecks. Use async processing for non-critical paths. "
                "Implement auto-scaling based on metrics. Consider data partitioning and denormalization for hot paths."
            ),
            "Design a concurrent job-processing queue that handles failures, retries, and priority levels.": (
                "Use a priority queue backed by Redis sorted sets or a message broker with priority topics. "
                "Workers pull jobs and claim ownership with distributed locks. "
                "Implement exponential backoff retries with a max retry count. "
                "Move failed jobs to a dead-letter queue for inspection. "
                "Track job state in a database. Use worker heartbeats to detect crashed workers and re-queue their jobs. "
                "Monitor queue depth and auto-scale workers."
            ),
            "Explain your approach to database sharding. What are the tradeoffs of range-based vs. hash-based sharding?": (
                "Sharding partitions data across multiple database instances to scale writes and storage. "
                "Range-based sharding assigns contiguous key ranges to shards enabling efficient range queries "
                "but risks hotspots when recent data is written to the same shard. "
                "Hash-based sharding distributes data uniformly avoiding hotspots "
                "but makes range queries expensive requiring scatter-gather across all shards. "
                "Use consistent hashing to minimize data movement when adding shards."
            ),
        },
    },
    "Data Scientist": {
        "fresher": {
            "What is overfitting and how can you detect it?": (
                "Overfitting occurs when a model memorizes training data including noise and fails to generalize. "
                "Detect it by comparing training accuracy to validation accuracy — a large gap indicates overfitting. "
                "Use learning curves and cross-validation. Solutions include regularization, more data, and simpler models."
            ),
            "Explain the bias-variance tradeoff in your own words.": (
                "Bias is error from oversimplified assumptions causing underfitting. "
                "Variance is error from sensitivity to small fluctuations in training data causing overfitting. "
                "Total error equals bias squared plus variance plus irreducible noise. "
                "The tradeoff means reducing one typically increases the other."
            ),
            "What is feature engineering? Give two examples.": (
                "Feature engineering creates new input features from raw data to improve model performance. "
                "Example one: extracting hour and day-of-week from a timestamp for demand forecasting. "
                "Example two: creating a price-per-square-foot feature for housing price prediction."
            ),
            "What is k-fold cross-validation and why is it used?": (
                "K-fold cross-validation splits data into k equal folds trains on k-1 folds and tests on the remaining fold "
                "rotating through all folds and averaging results. "
                "It gives a more reliable performance estimate than a single train-test split and helps detect overfitting."
            ),
            "What is the difference between linear and logistic regression?": (
                "Linear regression predicts a continuous numerical output by fitting a line through data. "
                "Logistic regression predicts a binary categorical outcome using the sigmoid function to output probabilities. "
                "Linear minimizes MSE while logistic maximizes log-likelihood."
            ),
        },
        "mid": {
            "You deployed a model with 95% training accuracy but only 60% on production. What do you investigate?": (
                "This indicates overfitting or distribution shift. "
                "Check if production data has different statistical properties than training data. "
                "Review the feature pipeline for data leakage in training. "
                "Inspect for concept drift if the relationship between features and target has changed. "
                "Retrain with more representative data and implement monitoring for prediction drift."
            ),
            "Compare L1 and L2 regularization. When would you use each?": (
                "L1 Lasso adds the sum of absolute weights producing sparse models by shrinking irrelevant features to zero — useful for feature selection. "
                "L2 Ridge adds the sum of squared weights shrinking all weights proportionally but rarely to zero — better when all features are relevant. "
                "Use L1 when many features are irrelevant. Use L2 when features are correlated. Elastic Net combines both."
            ),
            "What techniques do you use for feature selection on a dataset with 500+ features?": (
                "Filter methods: correlation matrix to remove highly correlated features, variance threshold to drop near-zero variance. "
                "Wrapper methods: recursive feature elimination with cross-validation. "
                "Embedded methods: Lasso regularization, tree-based feature importance from Random Forest or XGBoost. "
                "Dimensionality reduction: PCA for numeric features. Validate on held-out set to avoid selection bias."
            ),
            "How does stratified k-fold differ from regular k-fold and when does it matter?": (
                "Stratified k-fold ensures each fold preserves the original class proportion. "
                "Regular k-fold may produce folds with imbalanced class distributions by chance. "
                "Stratified k-fold matters for imbalanced datasets where one class is rare "
                "as without stratification some folds may contain no minority class examples."
            ),
            "Compare Ridge, Lasso, and Elastic Net. What problem does each solve?": (
                "Ridge L2 prevents overfitting when features are correlated by penalizing large weights without eliminating features. "
                "Lasso L1 performs automatic feature selection by zeroing irrelevant feature weights for sparse problems. "
                "Elastic Net combines L1 and L2 — it selects features like Lasso while handling correlated feature groups like Ridge."
            ),
        },
        "senior": {
            "Design a production ML pipeline from data ingestion to model monitoring. What are the failure points?": (
                "Pipeline stages: data ingestion from sources, validation with Great Expectations, feature engineering and feature store, "
                "model training with experiment tracking in MLflow and model registry, "
                "canary deployment behind feature flags, real-time serving, "
                "monitoring for data drift prediction drift and latency. "
                "Failure points: schema changes in upstream data, training-serving skew, silent prediction degradation, feature store staleness."
            ),
            "Explain how gradient boosting works and compare XGBoost, LightGBM, and CatBoost for a tabular dataset.": (
                "Gradient boosting builds an ensemble of weak learners sequentially where each tree corrects residual errors of the previous. "
                "XGBoost introduced regularization and level-wise tree growth. "
                "LightGBM uses leaf-wise growth and histogram-based splitting making it faster on large datasets. "
                "CatBoost handles categorical features natively with ordered boosting reducing overfitting. "
                "For large tabular datasets LightGBM is fastest, CatBoost best for many categoricals."
            ),
            "How would you handle automated feature engineering on a high-cardinality categorical dataset at scale?": (
                "Use target encoding with cross-validation folds to prevent leakage. "
                "Apply frequency encoding and embedding layers for neural approaches. "
                "Use Featuretools for automated deep feature synthesis. "
                "For time-series categoricals compute rolling aggregates. "
                "Reduce cardinality by grouping rare categories. "
                "Store engineered features in a feature store for reuse. Monitor feature drift in production."
            ),
            "How do you apply cross-validation correctly for time-series data? What leakage pitfalls must you avoid?": (
                "Use TimeSeriesSplit which respects temporal order — train on past validate on future. "
                "Never shuffle time-series data before splitting. "
                "Avoid leakage from future-looking lag features computed on the full dataset before splitting. "
                "Use purging to remove training samples whose labels overlap with the validation period. "
                "Use embargoing to block samples immediately before the test period."
            ),
            "Describe an end-to-end approach for building and deploying a multi-output regression model in a real-time system.": (
                "Use multi-task learning if outputs are correlated to share feature representations. "
                "Evaluate with per-output RMSE and a composite metric. "
                "Serve with a REST endpoint returning a vector of predictions. "
                "Monitor each output independently for drift. "
                "Use async batching for throughput and caching for repeated identical inputs."
            ),
        },
    },
    "AI/ML Engineer": {
        "fresher": {
            "What is gradient descent and what problem does it solve?": (
                "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively updating parameters "
                "in the direction of the negative gradient. It solves the problem of finding optimal model weights during training. "
                "Learning rate controls the step size. Variants include batch stochastic and mini-batch gradient descent."
            ),
            "Explain what a neural network is, including layers and activation functions.": (
                "A neural network consists of an input layer hidden layers and an output layer. "
                "Neurons are connected by weights. Activation functions like ReLU sigmoid and softmax introduce non-linearity "
                "allowing the network to learn complex patterns. "
                "Training adjusts weights using backpropagation and gradient descent."
            ),
            "What are Large Language Models (LLMs) and how are they trained?": (
                "LLMs are transformer-based neural networks trained on massive text datasets to understand and generate language. "
                "They learn by predicting the next token using self-supervised learning. "
                "Fine-tuning adapts them to specific tasks. Examples include GPT BERT and Claude. "
                "They use self-attention to capture long-range dependencies between words."
            ),
            "What is backpropagation and why is it important?": (
                "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule backwards through the network. "
                "The forward pass computes predictions and loss. The backward pass propagates gradients from output to input. "
                "These gradients are used by gradient descent to update weights and reduce loss."
            ),
            "What metrics would you use to evaluate a classification model?": (
                "Accuracy measures overall correctness. Precision is true positives divided by predicted positives. "
                "Recall is true positives divided by actual positives. F1-score is the harmonic mean of precision and recall. "
                "AUC-ROC measures discrimination ability across thresholds. "
                "For imbalanced classes F1 and AUC are more informative than accuracy."
            ),
        },
        "mid": {
            "Compare Adam, SGD, and RMSprop optimizers. When would you pick each?": (
                "SGD with momentum generalizes well for image models but requires careful learning rate tuning. "
                "RMSprop adapts the learning rate per parameter using a moving average of squared gradients — good for RNNs. "
                "Adam combines momentum and adaptive learning rates making it robust for most tasks. "
                "Use Adam as default, SGD with cosine annealing for computer vision, RMSprop for recurrent architectures."
            ),
            "When would you choose a CNN over an RNN and vice versa? Give use cases.": (
                "CNNs excel at spatially structured data like images and short text using local feature detectors via convolution. "
                "RNNs LSTMs GRUs capture sequential dependencies suited for variable-length sequences like time-series and language. "
                "Use CNN for image classification, object detection, short document classification. "
                "Use RNN for speech recognition, translation, and time-series forecasting. "
                "Transformers have largely replaced RNNs for NLP."
            ),
            "Explain the difference between fine-tuning an LLM and using Retrieval-Augmented Generation (RAG).": (
                "Fine-tuning updates model weights on domain-specific data teaching new knowledge — requires labelled data and compute. "
                "RAG keeps the base model frozen and retrieves relevant documents at inference time injecting them into the prompt. "
                "RAG is cheaper more up-to-date and traceable. Fine-tuning is better for style adaptation. "
                "Often combine both: fine-tune for format and tone, RAG for factual grounding."
            ),
            "How do you address the vanishing and exploding gradient problems?": (
                "Vanishing gradients: use ReLU activations apply batch normalization use residual connections use LSTMs in recurrent networks. "
                "Exploding gradients: apply gradient clipping to cap gradient norm use weight regularization reduce learning rate. "
                "Both: use careful weight initialization like Xavier or He initialization and layer normalization in transformers."
            ),
            "Explain the precision-recall tradeoff. How do you decide where to set the classification threshold?": (
                "Precision is the fraction of positive predictions that are correct. Recall is the fraction of actual positives detected. "
                "Lowering the threshold increases recall but decreases precision. "
                "Use precision-recall curve and F1 score to find the balance. "
                "For fraud detection prioritize recall. For spam filtering prioritize precision. "
                "Choose threshold based on the business cost of false positives vs false negatives."
            ),
        },
        "senior": {
            "Design an end-to-end MLOps pipeline for training, serving, and monitoring a recommendation model at 100M users.": (
                "Data pipeline ingests user events via Kafka into a feature store with batch and real-time features. "
                "Training pipeline runs in Kubeflow or SageMaker Pipelines with experiment tracking in MLflow and model registry. "
                "Serving via two-stage retrieval: candidate generation with ANN search Faiss then re-ranking with a lightweight model. "
                "Deploy with canary releases behind feature flags. "
                "Monitor feature drift prediction distribution click-through rate and latency. "
                "Auto-retrain on drift detection."
            ),
            "Explain the transformer architecture in depth — attention mechanisms, positional encoding, and why it replaced RNNs.": (
                "Transformers use self-attention to compute relationships between all token pairs in parallel enabling full context access. "
                "Scaled dot-product attention computes query key value matrices with softmax normalization. "
                "Multi-head attention runs multiple attention heads capturing different relationship types. "
                "Positional encoding injects sequence order information since attention is permutation-invariant. "
                "Replaced RNNs because they parallelize over sequence length and handle long-range dependencies better."
            ),
            "How would you deploy a 70B parameter LLM to serve low-latency inference requests? Discuss quantization, batching, and hardware.": (
                "Quantize to INT8 or INT4 using bitsandbytes or GPTQ to reduce memory footprint. "
                "Use tensor parallelism across multiple A100s or H100s via vLLM or Megatron. "
                "Implement continuous batching to maximize GPU utilization. "
                "Use paged attention in vLLM to avoid KV cache fragmentation. "
                "Deploy with a load balancer and auto-scaling. Cache repeated prompts. Target P99 latency under 2 seconds via token streaming."
            ),
            "What techniques do you use to ensure stable training of large models? Cover learning rate schedules, gradient clipping, and mixed precision.": (
                "Use warmup then cosine decay learning rate schedule to avoid instability at the start. "
                "Apply gradient clipping with a global norm threshold of 1.0 to prevent exploding gradients. "
                "Train with BF16 mixed precision — more numerically stable than FP16 for large models. "
                "Use gradient accumulation for effective large batch sizes without memory overflow. "
                "Monitor gradient norms and loss curves. Checkpoint frequently."
            ),
            "How do you design an A/B testing framework to safely roll out a new ML model in production?": (
                "Define hypothesis primary metric and minimum detectable effect before starting. "
                "Randomly split users into control and treatment groups ensuring no contamination. "
                "Run a power analysis to determine required sample size and test duration. "
                "Monitor guardrail metrics like latency error rate and revenue alongside the primary metric. "
                "Use sequential testing or Bayesian methods for early stopping. "
                "After statistical significance deploy the winner with a gradual ramp while monitoring for novelty effects."
            ),
        },
    },
    "Product Manager": {
        "fresher": {
            "What is an MVP and why is it important?": (
                "Minimum Viable Product is the simplest version of a product that delivers core value and allows validating hypotheses with real users. "
                "It minimizes waste by building only what is needed to learn. "
                "Follows lean startup principles: build measure learn. Enables fast iteration based on real user feedback."
            ),
            "How do you prioritize a list of feature requests?": (
                "Use frameworks like MoSCoW Must Should Could Won't, RICE Reach Impact Confidence Effort, or ICE Impact Confidence Ease. "
                "Consider user value business impact technical feasibility and strategic alignment. "
                "Talk to users and stakeholders use data on usage and pain points and balance short-term vs long-term goals."
            ),
            "Describe the four stages of the product lifecycle.": (
                "Introduction: product launches with high marketing spend and low sales. "
                "Growth: rapid user adoption and revenue increase focus on scaling. "
                "Maturity: sales peak competition intensifies focus on differentiation and retention. "
                "Decline: sales fall decision to innovate pivot or discontinue."
            ),
            "What metrics would you use to measure whether a product feature is successful?": (
                "Define success metrics before launch. Use adoption rate engagement frequency retention impact and conversion rate. "
                "Track NPS for satisfaction. Monitor error rates and performance metrics. "
                "Use AARRR framework: Acquisition Activation Retention Revenue Referral. Compare against baseline using A/B testing."
            ),
            "How do you manage a stakeholder who disagrees with your product decision?": (
                "Listen actively to understand their concern. Share the data and user research behind your decision. "
                "Acknowledge valid points and look for compromise. Align on shared goals. "
                "Escalate to leadership only after exhausting direct resolution."
            ),
        },
        "mid": {
            "You have 10 feature requests and capacity for 3 this quarter. Walk me through how you decide.": (
                "Start by mapping each request to strategic goals. Score each using RICE: Reach Impact Confidence Effort. "
                "Factor in dependencies and technical debt. Validate top candidates with user interviews. "
                "Present the shortlist to engineering for feasibility and to leadership for alignment. "
                "Communicate the rationale for what was not selected to build trust with stakeholders."
            ),
            "A key stakeholder wants a feature your data shows will hurt retention. How do you handle this?": (
                "Present the data clearly showing the retention analysis and predicted impact. "
                "Understand the stakeholder's underlying goal — there may be a better solution. "
                "Propose running a small A/B test to validate both hypotheses before full commitment. "
                "If they still push forward document the decision and agreed success criteria."
            ),
            "Your product is in the maturity stage with declining growth. What strategies do you consider?": (
                "Expand to new user segments or geographies. Deepen engagement with power users through advanced features. "
                "Bundle with complementary products. Improve retention through personalization and better onboarding. "
                "Reduce cost to serve with automation. Explore adjacent use cases for the core technology."
            ),
            "How do you write OKRs for a product team? Give an example for a SaaS product.": (
                "OKRs consist of an inspiring Objective and 3-5 measurable Key Results. "
                "Objective: Grow user engagement meaningfully this quarter. "
                "KR1: Increase DAU/MAU ratio from 30% to 40%. "
                "KR2: Reduce 7-day churn from 25% to 18%. "
                "KR3: Increase median session length from 4 to 6 minutes."
            ),
            "How do you align engineering, design, and marketing around a product roadmap?": (
                "Share a single source-of-truth roadmap with clear goals not just a list of features. "
                "Run joint discovery sessions involving all disciplines early. "
                "Use a RACI matrix to clarify responsibilities. "
                "Hold regular cross-functional syncs and a monthly roadmap review. Escalate blockers quickly."
            ),
        },
        "senior": {
            "How do you decide whether to build, buy, or partner for a new capability?": (
                "Build when the capability is core differentiation and you have engineering capacity. "
                "Buy when there is a mature solution and speed to market outweighs customization — evaluate TCO and integration risk. "
                "Partner when the capability is adjacent and a partner brings distribution or complementary expertise. "
                "Factors: strategic importance time to market cost integration complexity vendor lock-in and team bandwidth."
            ),
            "You're entering a new market with strong incumbents. How do you define a product strategy?": (
                "Start with a wedge — find an underserved segment the incumbents ignore. "
                "Build a focused product that nails a specific job-to-be-done better than incumbents. "
                "Use the beachhead to expand adjacently once you have a loyal base. "
                "Compete on speed simplicity or price rather than feature parity. "
                "Define a clear differentiated positioning and validate with ICP interviews before building."
            ),
            "How do you manage a portfolio of products at different lifecycle stages with limited resources?": (
                "Categorize products by lifecycle stage and growth potential. "
                "Invest heavily in growth-stage products, maintain maturity-stage products efficiently, "
                "and sunset declining products that drain resources. "
                "Use portfolio-level OKRs to prevent local optimization. "
                "Rebalance quarterly based on market signals."
            ),
            "How do you drive organisational alignment when your product vision challenges the status quo?": (
                "Build a compelling narrative grounded in customer pain and market opportunity. "
                "Win early adopters in leadership by showing quick wins that validate the direction. "
                "Create a cross-functional coalition to build momentum bottom-up and top-down simultaneously. "
                "Address loss aversion by showing what the organization gains. Use data and external benchmarks."
            ),
            "Describe a time you had to lead through ambiguity with incomplete data. How did you structure your decision?": (
                "Frame the decision by identifying what is known, what is unknown, and the cost of delay vs a wrong decision. "
                "Run the fastest possible experiments to reduce key uncertainties. "
                "Make a reversible decision when possible and set clear triggers to revisit. "
                "Communicate the uncertainty and reasoning transparently. Document assumptions explicitly."
            ),
        },
    },
    "Finance Analyst": {
        "fresher": {
            "What is Net Present Value (NPV) and what does a positive NPV mean?": (
                "NPV calculates the present value of future cash flows discounted at the cost of capital minus the initial investment. "
                "A positive NPV means the investment generates more value than its cost and creates shareholder value. "
                "A negative NPV means destroy value. It is the primary tool for capital budgeting decisions."
            ),
            "What does EBITDA measure and why do analysts use it?": (
                "EBITDA is Earnings Before Interest Taxes Depreciation and Amortization. "
                "It measures operating profitability independent of capital structure tax jurisdiction and accounting policies. "
                "Analysts use it to compare companies across industries and as a proxy for operating cash flow. "
                "It is also used in EV/EBITDA valuation multiples."
            ),
            "Explain portfolio diversification and why it reduces risk.": (
                "Diversification spreads investments across assets that are not perfectly correlated. "
                "When one asset falls uncorrelated assets may rise reducing overall portfolio volatility. "
                "Unsystematic risk specific to individual companies can be eliminated through diversification. "
                "Systematic market risk cannot be diversified away. "
                "Modern Portfolio Theory shows an efficient frontier of portfolios with optimal risk-return tradeoffs."
            ),
            "What are the main types of financial risk and how do you identify them?": (
                "Market risk from price movements in equities rates and commodities. "
                "Credit risk from counterparty default. "
                "Liquidity risk from inability to sell assets quickly. "
                "Operational risk from process failures or fraud. "
                "Identify through financial statement analysis scenario analysis stress testing and risk registers."
            ),
            "What is ROI and how is it calculated?": (
                "Return on Investment equals net profit divided by cost of investment multiplied by 100. "
                "It measures the efficiency and profitability of an investment. "
                "Higher ROI is better. Limitations include ignoring time value of money and risk. "
                "Used to compare different investment options on a simple basis."
            ),
        },
        "mid": {
            "When would you use IRR instead of NPV for a capital budgeting decision? What are IRR's limitations?": (
                "Use IRR when comparing projects of different scales or presenting a percentage return to stakeholders. "
                "IRR is the discount rate that makes NPV zero. Accept if IRR exceeds the cost of capital. "
                "Limitations: multiple IRRs when cash flows change sign more than once, "
                "assumes reinvestment at the IRR rate which is unrealistic, "
                "cannot rank mutually exclusive projects by size. Modified IRR MIRR addresses the reinvestment assumption."
            ),
            "A company reports EBITDA of $50M but has heavy capex. What adjustments do you make for a cleaner valuation?": (
                "EBITDA ignores capex which is a real cash outflow. Calculate EBITDA minus capex for a maintenance capital-adjusted figure. "
                "Also adjust for non-recurring items restructuring charges one-off gains stock-based compensation. "
                "Normalize working capital movements. "
                "Consider lease-adjusted EBITDA under IFRS 16. "
                "Compare adjusted EBITDA margin to peers to assess operational efficiency."
            ),
            "How do you apply Modern Portfolio Theory to construct an efficient portfolio for a client?": (
                "Estimate expected returns standard deviations and pairwise correlations for each asset class. "
                "Use mean-variance optimization to find portfolios on the efficient frontier maximizing return for each level of risk. "
                "Identify the tangency portfolio maximizing Sharpe ratio. "
                "Adjust based on the client's risk tolerance investment horizon and liquidity needs. "
                "Rebalance periodically when allocations drift from targets."
            ),
            "Explain Value at Risk (VaR) and its limitations. How does Conditional VaR improve on it?": (
                "VaR estimates the maximum loss not exceeded at a given confidence level over a time horizon. "
                "Limitations: ignores losses beyond the threshold assumes normal distributions poor for tail risk not sub-additive. "
                "Conditional VaR CVaR or Expected Shortfall is the expected loss given that the loss exceeds VaR. "
                "CVaR better captures tail risk and is coherent making it preferred by regulators."
            ),
            "Compare ROI and ROIC. Why might ROIC be a better measure of business quality?": (
                "ROI is a general measure of any investment return. "
                "ROIC Return on Invested Capital measures operating profit NOPAT divided by total invested capital. "
                "ROIC reflects how efficiently management deploys capital. "
                "A company with ROIC above its WACC creates value; below WACC it destroys value. "
                "ROIC adjusts for capital structure making comparisons across companies more meaningful."
            ),
        },
        "senior": {
            "Walk me through the three main valuation approaches in an M&A transaction and when each is most appropriate.": (
                "DCF analysis discounts projected free cash flows at WACC plus terminal value — most appropriate when cash flows are predictable. "
                "Comparable company analysis uses public market trading multiples EV/EBITDA P/E — sets a market-clearing reference and reflects current sentiment. "
                "Precedent transaction analysis uses multiples paid in prior M&A deals — captures control premium and is most relevant for pricing an acquisition. "
                "A robust fairness opinion uses all three and triangulates within a valuation range."
            ),
            "How do you build an EBITDA bridge analysis when two companies with different accounting policies are being compared?": (
                "Restate both companies to the same accounting policies: align depreciation methods capitalize vs expense R&D consistently normalize lease treatment. "
                "Build a waterfall bridge showing starting EBITDA, add-back non-recurring items restructuring M&A costs, "
                "adjust for accounting policy differences run-rate synergies and pro-forma adjustments for acquisitions closed. "
                "Document every adjustment with rationale and magnitude for due diligence scrutiny."
            ),
            "Describe how you would construct a factor-based equity portfolio. What factors would you use and why?": (
                "Select well-researched factors: value low P/B P/E, quality high ROE low accruals, momentum prior 12-month returns, "
                "low volatility low beta, and size small cap premium. "
                "Construct factor scores standardize cross-sectionally and combine into a composite score. "
                "Use portfolio optimization with factor exposure constraints and transaction cost modeling. "
                "Monitor factor crowding and correlation between factors. Rebalance monthly or quarterly."
            ),
            "Design an enterprise risk management framework for a multinational company exposed to FX, credit, and liquidity risk.": (
                "Governance: risk committee with board oversight clear risk appetite statement and three lines of defense model. "
                "FX risk: identify natural hedges implement forward contracts and options for transactional exposure use net investment hedges for translation. "
                "Credit risk: counterparty limits based on credit ratings netting agreements collateral management CDS for large exposures. "
                "Liquidity risk: maintain liquidity buffer diversify funding sources stress test under multiple scenarios. "
                "Consolidated risk reporting with VaR CVaR and scenario analysis to the board quarterly."
            ),
            "A PE-backed company is planning an exit. How do you evaluate the return (IRR, MOIC, DPI) and what levers improve it?": (
                "IRR measures the annualized return accounting for time value — sensitive to hold period. "
                "MOIC money-on-invested-capital is total value returned divided by invested capital — measures absolute multiple. "
                "DPI distributions to paid-in measures realised returns providing liquidity certainty. "
                "Levers: revenue growth through organic expansion or add-on acquisitions margin improvement through operational efficiency "
                "multiple expansion via strategic positioning and optimal capital structure to minimize WACC. "
                "Exit route analysis: strategic sale typically highest multiple IPO for large platforms secondary buyout when fundamentals support."
            ),
        },
    },
    "Frontend Developer": {
        "fresher": {
            "What is the difference between HTML, CSS, and JavaScript?": (
                "HTML HyperText Markup Language defines the structure and content of a web page using elements like headings paragraphs and links. "
                "CSS Cascading Style Sheets controls the visual presentation — colours fonts layout and responsiveness. "
                "JavaScript adds interactivity and behaviour — handling events fetching data and updating the DOM dynamically. "
                "They work together: HTML is the skeleton CSS is the skin and JavaScript is the muscles."
            ),
            "Explain the CSS box model.": (
                "Every HTML element is a rectangular box consisting of four layers from inside out: content padding border and margin. "
                "Content is the actual text or image area. Padding is transparent space between content and border. "
                "Border surrounds the padding. Margin is the transparent space outside the border separating it from other elements. "
                "By default box-sizing is content-box so width applies only to content. Setting box-sizing border-box includes padding and border in the width making layouts easier."
            ),
            "What is the DOM and how does JavaScript interact with it?": (
                "The Document Object Model is a tree-like in-memory representation of an HTML document that the browser builds when a page loads. "
                "Each HTML element becomes a node in the tree. JavaScript can read and modify the DOM via APIs like document.getElementById querySelector innerHTML and addEventListener. "
                "Changing the DOM updates the page visually without a full reload. "
                "Excessive DOM manipulation is slow so frameworks like React use a virtual DOM to batch and minimise real DOM updates."
            ),
            "What is the difference between flexbox and CSS Grid?": (
                "Flexbox is a one-dimensional layout model — it arranges items along a single axis either row or column. "
                "It excels at distributing space and aligning items within a container along that axis. "
                "CSS Grid is two-dimensional — it handles rows and columns simultaneously making it ideal for overall page layouts. "
                "Use flexbox for component-level alignment like navigation bars and button groups. "
                "Use Grid for full-page or section-level layouts. They can be combined."
            ),
            "What is a JavaScript Promise?": (
                "A Promise is an object representing the eventual completion or failure of an asynchronous operation. "
                "It has three states: pending fulfilled or rejected. "
                "Use .then() to handle success and .catch() to handle errors. "
                "Promises chain cleanly avoiding callback hell. "
                "async/await is syntactic sugar built on top of Promises making asynchronous code read synchronously. "
                "Common uses include fetch API calls file reading and timers."
            ),
        },
        "mid": {
            "Explain React hooks — useState, useEffect, and useContext with use cases.": (
                "useState adds local component state — returns a state value and a setter function. Use it for form inputs toggles and counters. "
                "useEffect runs side effects after render — data fetching subscriptions and DOM manipulation. "
                "Return a cleanup function to avoid memory leaks. The dependency array controls when it reruns. "
                "useContext consumes a React context value avoiding prop drilling. "
                "Combine all three to build a self-contained component that fetches data manages loading state and reads global theme or auth context."
            ),
            "How do you optimize the performance of a React application?": (
                "Avoid unnecessary re-renders using React.memo for pure components and useMemo/useCallback for expensive computations and stable function references. "
                "Use code splitting with React.lazy and Suspense to load components on demand reducing initial bundle size. "
                "Virtualise long lists with react-window to render only visible rows. "
                "Profile with React DevTools Profiler to find expensive renders. "
                "Optimise images with next/image or lazy loading. Move heavy state down the tree to limit re-render scope."
            ),
            "Compare Redux, Zustand, and React Context for state management. When do you use each?": (
                "React Context is built-in best for low-frequency global data like theme or auth — causes full subtree re-renders on change so unsuitable for high-frequency updates. "
                "Redux is a predictable state container with middleware support time-travel debugging and the Redux DevTools — ideal for complex apps with many state interactions. "
                "Zustand is a lightweight alternative with minimal boilerplate a simple hook-based API and fine-grained subscriptions preventing unnecessary re-renders. "
                "Use Context for simple global config Redux for complex enterprise apps Zustand for most mid-sized apps."
            ),
            "What is web accessibility (WCAG) and how do you implement it?": (
                "WCAG Web Content Accessibility Guidelines define standards to make web content accessible to people with disabilities. "
                "Four principles: Perceivable Operable Understandable and Robust POUR. "
                "Implementation: use semantic HTML elements nav main article, provide alt text on images, ensure keyboard navigability with visible focus states, "
                "maintain sufficient colour contrast ratio of at least 4.5:1 for normal text, use ARIA roles labels and live regions only when HTML semantics are insufficient, "
                "and test with screen readers like NVDA or VoiceOver."
            ),
            "Describe a CSS architecture strategy (BEM, CSS Modules, CSS-in-JS) and its tradeoffs.": (
                "BEM Block Element Modifier is a naming convention for plain CSS — clear predictable class names but verbose and requires team discipline. "
                "CSS Modules scope class names locally by default at build time eliminating naming collisions with minimal runtime overhead. "
                "CSS-in-JS libraries like styled-components or Emotion colocate styles with components enabling dynamic theming and prop-driven styles but add runtime overhead and bundle size. "
                "For large teams CSS Modules offer the best balance of scoping and performance. CSS-in-JS suits design-system-driven component libraries."
            ),
        },
        "senior": {
            "How would you architect a micro-frontend system for a large e-commerce platform?": (
                "Split the UI by business domain: product catalogue checkout account and recommendations each owned by a separate team. "
                "Use Module Federation in Webpack 5 to allow each micro-frontend to be deployed independently and loaded at runtime by a shell application. "
                "Define a shared dependency strategy to avoid duplicate React instances. "
                "Establish a design system as a shared npm package for UI consistency. "
                "Use single-spa or a custom orchestrator for routing. "
                "Each micro-frontend has its own CI/CD pipeline. Share auth state via a global event bus or shared session store."
            ),
            "Compare SSR, SSG, ISR, and CSR rendering strategies. When do you choose each in Next.js?": (
                "CSR Client-Side Rendering ships a minimal HTML shell and renders in the browser — fast subsequent navigations but poor SEO and slow initial load. "
                "SSG Static Site Generation pre-renders at build time — fastest delivery via CDN ideal for marketing pages and blogs that rarely change. "
                "SSR Server-Side Rendering generates HTML on each request — good for personalised pages with SEO requirements at the cost of server latency. "
                "ISR Incremental Static Regeneration revalidates static pages in the background on a schedule — best of SSG and SSR for content that changes infrequently."
            ),
            "How do you design and maintain a component library used by 10+ engineering teams?": (
                "Build on a design token system defining colours spacing and typography as variables. "
                "Document every component with Storybook including prop tables accessibility notes and usage guidelines. "
                "Publish to a private npm registry with semantic versioning. "
                "Use a monorepo with Turborepo or Nx for co-locating the library with consuming apps. "
                "Establish an RFC process for new components and breaking changes. "
                "Automate visual regression testing with Chromatic. Maintain a changelog and migration guides for major versions."
            ),
            "Walk me through optimizing Core Web Vitals (LCP, INP, CLS) for a content-heavy site.": (
                "LCP Largest Contentful Paint: preload the hero image with <link rel=preload>, serve images in WebP/AVIF from a CDN, use priority prop in next/image. "
                "INP Interaction to Next Paint replaces FID: reduce long tasks by code-splitting and deferring non-critical JS, use web workers for heavy computation, optimise event handlers. "
                "CLS Cumulative Layout Shift: always specify width and height on images and ads, avoid inserting DOM above existing content, use font-display swap with size-adjust. "
                "Measure with Lighthouse PageSpeed Insights and real user monitoring via CrUX."
            ),
            "What are the key frontend security threats (XSS, CSRF, CSP) and how do you mitigate them?": (
                "XSS Cross-Site Scripting: sanitise all user input never use innerHTML with untrusted data use textContent instead and apply a strict Content-Security-Policy header. "
                "CSRF Cross-Site Request Forgery: use SameSite=Strict or Lax cookies validate CSRF tokens on state-changing requests. "
                "CSP Content-Security-Policy header restricts which scripts styles and resources can load preventing injected scripts from executing. "
                "Also: avoid storing sensitive data in localStorage use HTTPS only set HttpOnly and Secure flags on session cookies and audit third-party dependencies regularly."
            ),
        },
    },
    "Backend Developer": {
        "fresher": {
            "What is the difference between REST and GraphQL?": (
                "REST Representational State Transfer uses fixed URL endpoints each returning a predefined data shape — simple cacheable and well-understood. "
                "GraphQL uses a single endpoint where the client specifies exactly what fields it needs in a query eliminating over-fetching and under-fetching. "
                "REST is better for simple CRUD APIs and public APIs. "
                "GraphQL excels when clients have diverse data needs such as mobile and web consuming different subsets of the same data. "
                "GraphQL adds complexity in caching and security requiring query depth limiting and cost analysis."
            ),
            "What is a database transaction and what are ACID properties?": (
                "A transaction is a sequence of database operations executed as a single unit — either all succeed or all fail. "
                "ACID: Atomicity means all operations in the transaction complete or none do. "
                "Consistency ensures the database moves from one valid state to another respecting constraints. "
                "Isolation ensures concurrent transactions do not interfere with each other. "
                "Durability guarantees committed transactions survive system failures. "
                "ACID is critical for financial operations where partial updates would leave data in an inconsistent state."
            ),
            "Explain the difference between authentication and authorization.": (
                "Authentication verifies who you are — confirming identity via passwords tokens or biometrics. "
                "Authorization determines what you are allowed to do — checking permissions after identity is established. "
                "Example: logging in is authentication accessing an admin panel only if you have the admin role is authorization. "
                "Common implementations: JWT for stateless auth OAuth 2.0 for delegated access RBAC Role-Based Access Control for authorization."
            ),
            "What is caching and why is it used?": (
                "Caching stores frequently accessed data in fast storage like memory to reduce latency and backend load. "
                "Instead of hitting a slow database or external API on every request the app returns a cached response. "
                "Types: in-memory cache like Redis or Memcached CDN edge cache browser cache and application-level cache. "
                "Key considerations: cache invalidation strategy TTL time-to-live cache stampede prevention and data freshness requirements. "
                "Caching can reduce database load by 80-90% for read-heavy workloads."
            ),
            "What is the difference between SQL and NoSQL databases?": (
                "SQL databases are relational — data is stored in tables with fixed schemas and relationships enforced by foreign keys. "
                "They use structured query language and support ACID transactions. Examples: PostgreSQL MySQL. "
                "NoSQL databases are non-relational — flexible schemas document key-value column-family or graph models. "
                "They scale horizontally more easily and suit unstructured or rapidly changing data. Examples: MongoDB Redis Cassandra. "
                "Choose SQL for complex queries and data integrity; NoSQL for high write throughput and flexible schemas."
            ),
        },
        "mid": {
            "Design a rate limiter for an API that allows 100 requests per minute per user.": (
                "Use a sliding window counter algorithm in Redis. Store a sorted set per user keyed by timestamp. "
                "On each request add the current timestamp and remove entries older than 60 seconds. "
                "If the set size exceeds 100 reject the request with HTTP 429 Too Many Requests and a Retry-After header. "
                "For distributed systems use Redis atomic Lua scripts or INCR with EXPIRE for the fixed window variant. "
                "Expose rate limit headers X-RateLimit-Limit X-RateLimit-Remaining X-RateLimit-Reset. "
                "Apply limits at the API gateway layer to avoid burdening application servers."
            ),
            "How do you manage database connection pooling and why does it matter?": (
                "Opening a new database connection is expensive — it involves TCP handshake TLS auth and session setup. "
                "A connection pool maintains a reusable set of open connections shared across requests. "
                "Configure pool size based on database max_connections and application concurrency — a common formula is cores times 2 plus effective spindle count. "
                "Set idle timeout to reclaim stale connections. Monitor pool wait time and connection exhaustion. "
                "In Python use SQLAlchemy's pool_size max_overflow and pool_timeout. In microservices use PgBouncer for PostgreSQL to multiplex thousands of application connections onto fewer DB connections."
            ),
            "Compare monolithic and microservices architecture — when would you choose each?": (
                "A monolith is a single deployable unit — simple to develop test and deploy initially with low operational overhead. "
                "Best for small teams early-stage products and when domain boundaries are not yet clear. "
                "Microservices split the system into independently deployable services each owning its data store. "
                "Benefits: independent scaling independent deployment fault isolation and team autonomy. "
                "Costs: network latency distributed tracing complex deployments and eventual consistency challenges. "
                "Choose microservices when different components have clearly different scaling needs or when multiple teams need to deploy independently."
            ),
            "Compare JWT and session-based authentication. What are the security tradeoffs?": (
                "Session-based auth stores session state server-side in memory or a database and sends a session ID cookie to the client. "
                "Easy to invalidate by deleting the session but requires shared session storage in distributed systems. "
                "JWT JSON Web Token is stateless — the token itself carries claims and is verified with a secret or public key. "
                "Scales easily across services with no shared state but tokens cannot be individually invalidated before expiry without a blocklist. "
                "Security: JWTs must use short expiry times and secure refresh token rotation. Always use HTTPS set HttpOnly and Secure flags."
            ),
            "Explain the difference between horizontal and vertical scaling with examples.": (
                "Vertical scaling adds more CPU RAM or storage to an existing server — simple but has a hard ceiling and creates a single point of failure. "
                "Horizontal scaling adds more servers and distributes load across them — theoretically unlimited scale but requires stateless application design and a load balancer. "
                "Example: scaling a PostgreSQL primary vertically to handle more read load vs adding read replicas horizontally. "
                "Modern cloud-native apps prefer horizontal scaling with auto-scaling groups enabling elasticity and high availability."
            ),
        },
        "senior": {
            "Design a distributed task queue system that guarantees at-least-once delivery.": (
                "Use a message broker like RabbitMQ or SQS with durable queues and persistent messages. "
                "Consumers acknowledge messages only after successful processing. If a consumer crashes the broker redelivers the unacknowledged message. "
                "This guarantees at-least-once delivery so consumers must be idempotent — processing the same message twice must not cause duplicate side effects. "
                "Use a unique job ID and a processed-set in Redis or a database to deduplicate. "
                "Implement dead-letter queues for messages that fail repeatedly. Monitor queue depth and consumer lag for capacity planning."
            ),
            "How do you approach event-driven architecture with Kafka? Explain partitioning, consumers, and delivery guarantees.": (
                "Kafka stores events in topics divided into partitions for parallelism. Each partition is an ordered append-only log. "
                "Producers write to a partition determined by a key hash ensuring ordered delivery per key. "
                "Consumer groups share partitions — each partition is consumed by one consumer in the group enabling parallel processing. "
                "Kafka offers at-most-once at-least-once and exactly-once semantics via idempotent producers and transactional APIs. "
                "Retention is configurable allowing event replay for new consumers or recovery. "
                "Design topics around business events not RPC calls. Use schema registry with Avro for schema evolution."
            ),
            "Explain CQRS and Event Sourcing — when are they appropriate and what are the operational costs?": (
                "CQRS Command Query Responsibility Segregation separates write models Commands from read models Queries allowing each to be optimised independently. "
                "Reads use denormalised projections tuned for query performance. "
                "Event Sourcing stores state as an immutable sequence of events rather than current state — the current state is derived by replaying events. "
                "Appropriate when audit trails are required when you need temporal queries or when multiple read models must be derived from the same writes. "
                "Operational costs: eventual consistency complexity snapshot management event schema versioning and higher infrastructure and cognitive overhead."
            ),
            "How do you design observability into a backend system (logs, metrics, traces)?": (
                "The three pillars: structured logs metrics and distributed traces. "
                "Logs: emit structured JSON logs with trace ID request ID and context fields. Ship to a centralised log aggregator like Loki or Elasticsearch. "
                "Metrics: instrument with Prometheus client libraries exposing RED metrics Rate Errors Duration per endpoint. Alert on SLO breaches. "
                "Traces: add OpenTelemetry instrumentation to propagate trace context across service boundaries. Visualise in Jaeger or Tempo. "
                "Build dashboards that correlate logs traces and metrics for a single request. Define SLOs before building dashboards."
            ),
            "Design a multi-tenant SaaS backend. How do you handle data isolation, billing, and permissions?": (
                "Data isolation options: shared database with tenant_id column row-level security in PostgreSQL, separate schemas per tenant for stronger isolation, or separate databases for regulated enterprise customers. "
                "Permissions: implement RBAC with roles scoped to tenants. Every API request validates the tenant context from the JWT. "
                "Billing: capture usage events to a metering service like Stripe Metered Billing or a custom aggregation pipeline. "
                "Multi-tenancy in application layer: tenant middleware extracts tenant ID and scopes all ORM queries. "
                "Rate-limit per tenant. Provide tenant-level audit logs. Design onboarding to provision tenants automatically."
            ),
        },
    },
    "DevOps Engineer": {
        "fresher": {
            "What is CI/CD and why is it important?": (
                "CI Continuous Integration is the practice of merging code changes frequently into a shared branch and automatically running builds and tests. "
                "CD Continuous Delivery or Deployment extends this by automatically releasing validated changes to staging or production. "
                "Benefits: faster feedback loops fewer integration conflicts higher deployment frequency and reduced manual errors. "
                "Common tools: GitHub Actions GitLab CI Jenkins CircleCI for CI and ArgoCD Spinnaker or direct kubectl for CD."
            ),
            "What is Docker and what problem does containerisation solve?": (
                "Docker packages an application and all its dependencies into a container — a lightweight isolated runtime environment. "
                "It solves the it works on my machine problem by ensuring the same container image runs identically in development staging and production. "
                "Containers share the host OS kernel unlike VMs making them faster to start and more resource-efficient. "
                "Docker images are built from Dockerfiles and stored in registries like Docker Hub or ECR. "
                "Key concepts: image layer caching multi-stage builds volume mounts and port mapping."
            ),
            "What is Kubernetes and what role does it play in container orchestration?": (
                "Kubernetes K8s automates the deployment scaling and management of containerised applications. "
                "It schedules containers onto nodes in a cluster ensures desired replica counts restarts failed containers and distributes traffic via Services. "
                "Key objects: Deployment manages replica sets and rolling updates Service exposes pods ConfigMap and Secret for configuration Ingress for HTTP routing. "
                "Kubernetes provides self-healing auto-scaling horizontal pod autoscaler and declarative configuration making it the standard for production container workloads."
            ),
            "What is Infrastructure as Code (IaC) and what tools are used?": (
                "IaC manages and provisions infrastructure through machine-readable configuration files rather than manual processes. "
                "This enables version-controlled reproducible and reviewable infrastructure changes. "
                "Declarative tools like Terraform and Pulumi describe desired state and automatically determine changes needed. "
                "Imperative tools like Ansible execute step-by-step configuration. "
                "AWS CloudFormation and Azure ARM templates are cloud-native IaC options. "
                "Benefits: consistency drift detection automated rollback and collaboration via pull requests."
            ),
            "What is a load balancer and how does it work?": (
                "A load balancer distributes incoming network traffic across multiple backend servers to prevent any single server from becoming a bottleneck. "
                "Layer 4 load balancers route at the TCP/UDP level based on IP and port. "
                "Layer 7 load balancers operate at the HTTP level routing based on URLs headers or cookies enabling more intelligent routing. "
                "Algorithms: round-robin least connections IP hash and weighted routing. "
                "Load balancers also perform health checks removing unhealthy backends from rotation. "
                "Examples: AWS ALB/NLB Nginx HAProxy."
            ),
        },
        "mid": {
            "Design a CI/CD pipeline for a microservices application with automated testing and blue-green deployment.": (
                "Pipeline stages: code push triggers linting and unit tests in parallel then integration tests against a test database then Docker image build and push to registry. "
                "Blue-green deployment: maintain two identical production environments blue live and green idle. "
                "Deploy new version to green run smoke tests then shift 100% of traffic via the load balancer to green. "
                "Blue becomes the rollback target. Use feature flags for gradual traffic shifting. "
                "Automate with GitHub Actions ArgoCD and Helm charts. "
                "Each microservice has its own pipeline but a shared release coordination step validates cross-service compatibility."
            ),
            "How do you manage secrets in a Kubernetes cluster securely?": (
                "Never store secrets in plaintext in YAML files or container images. "
                "Use HashiCorp Vault for centralised secret management with dynamic short-lived credentials. "
                "Alternatively use the cloud provider secret manager AWS Secrets Manager or GCP Secret Manager with the External Secrets Operator to sync secrets into Kubernetes. "
                "Enable envelope encryption for Kubernetes etcd at rest. "
                "Use Vault Agent or CSI driver to inject secrets as environment variables or mounted files. "
                "Rotate secrets automatically and audit access. Restrict secret access with RBAC at the namespace level."
            ),
            "What is your approach to container resource limits, autoscaling, and health checks in Kubernetes?": (
                "Set resource requests and limits on every container. Requests are used for scheduling limits prevent noisy neighbours. "
                "Set limits conservatively and tune based on profiling. Use LimitRanges to enforce defaults. "
                "HPA Horizontal Pod Autoscaler scales replicas based on CPU memory or custom metrics from Prometheus. "
                "VPA Vertical Pod Autoscaler recommends and sets right-sized requests automatically. "
                "Health checks: liveness probe restarts a crashed container readiness probe gates traffic to only ready pods startupProbe for slow-starting containers. "
                "Use httpGet or exec probes with appropriate initialDelaySeconds and periodSeconds."
            ),
            "How do you set up monitoring and alerting for a production system? What tools and SLO/SLA concepts apply?": (
                "Deploy Prometheus for metrics collection and Grafana for dashboards. Instrument applications with client libraries exposing RED metrics. "
                "Define SLOs Service Level Objectives: latency p99 under 200ms availability 99.9%. "
                "Create SLA Service Level Agreements with customers based on the SLO minus an error budget. "
                "Alert on error budget burn rate not raw thresholds to avoid alert fatigue. "
                "Use PagerDuty or Alertmanager for on-call routing. "
                "Ship logs to Loki or Elasticsearch. Add distributed tracing with OpenTelemetry and Jaeger for root cause analysis."
            ),
            "Explain Terraform state management, remote backends, and workspace strategies for multi-environment infrastructure.": (
                "Terraform state tracks the mapping between your configuration and real-world resources. "
                "Store state remotely in S3 with DynamoDB locking or Terraform Cloud to enable team collaboration and prevent concurrent modifications. "
                "Enable state encryption and versioning for auditability and rollback. "
                "Workspaces create isolated state files for different environments dev staging prod within the same configuration. "
                "Alternatively use separate directories or modules per environment for stronger isolation. "
                "Use .tfvars files for environment-specific variables and store sensitive values in a secrets manager not in state."
            ),
        },
        "senior": {
            "Design a multi-region, multi-cloud deployment strategy with automated failover for a 99.99% SLA service.": (
                "Active-active deployment across two regions with a global load balancer like AWS Route 53 or Cloudflare routing traffic based on latency or geolocation. "
                "Each region is fully self-sufficient with its own database replica application tier and cache. "
                "Use CockroachDB or Spanner for multi-region strongly consistent storage or DynamoDB Global Tables for eventual consistency. "
                "Failover: health checks trigger automatic DNS failover within 30 seconds. Chaos testing validates failover quarterly. "
                "Multi-cloud adds portability but increases operational complexity — use Terraform and container abstraction to remain cloud-agnostic. "
                "RPO and RTO targets must be defined to select the right replication strategy."
            ),
            "How do you build an internal developer platform that reduces cognitive load for engineering teams?": (
                "Start by interviewing engineering teams to identify their top pain points in deployment configuration and observability. "
                "Build a golden path: opinionated templates for new services scaffolding service creation with CI/CD observability and secrets pre-configured. "
                "Deploy a self-service portal using Backstage for service catalog ownership tracking and runbooks. "
                "Abstract Kubernetes complexity behind a simple deploy command or ArgoCD Application spec. "
                "Treat the platform as a product with an internal SLA dedicated roadmap and user feedback loop. "
                "Measure success by onboarding time to first deploy and reduction in platform-related support tickets."
            ),
            "Describe a cloud cost optimisation strategy for a company spending $500K/month on AWS.": (
                "Start with a cost visibility layer using AWS Cost Explorer and tagging enforced by SCPs across all accounts. "
                "Rightsizing: use AWS Compute Optimizer recommendations to downsize over-provisioned EC2 and RDS instances. "
                "Reserved Instances or Savings Plans for steady-state workloads — typically 30-40% savings. "
                "Spot Instances for fault-tolerant batch and ML workloads — up to 70% cheaper. "
                "Storage: move infrequently accessed S3 data to Glacier using lifecycle policies. Delete unattached EBS volumes and old snapshots. "
                "Architect for elasticity to scale down nights and weekends. Implement FinOps culture with team-level cost accountability."
            ),
            "How do you implement DevSecOps — integrating security scanning into every stage of the pipeline?": (
                "Shift left: integrate SAST static analysis into the IDE and pre-commit hooks with Semgrep or SonarQube. "
                "In CI: run SAST DAST dependency vulnerability scanning with Snyk or Dependabot and container image scanning with Trivy or Grype. "
                "IaC scanning with Checkov or Terrascan catches misconfigured infrastructure before deployment. "
                "Runtime: deploy Falco for container runtime threat detection. Use OPA Gatekeeper for admission control in Kubernetes. "
                "Secrets scanning: block hardcoded secrets with git-secrets or Trufflehog in pre-commit and CI. "
                "Build a security champion model — embed security into each engineering team."
            ),
            "What is chaos engineering and how would you implement a game-day program for a critical production system?": (
                "Chaos engineering is the practice of deliberately injecting failures into a production or staging system to uncover weaknesses before they cause incidents. "
                "Start with a hypothesis: if we kill one of the three API replicas the service continues serving requests with under 1% error rate. "
                "Use Chaos Monkey LitmusChaos or AWS Fault Injection Simulator to inject CPU spikes network latency partition or pod kills. "
                "Run game days quarterly with a defined blast radius and a rollback plan. Monitor observability dashboards in real time. "
                "Document every finding update runbooks and fix the most critical weaknesses. "
                "Build a steady-state metric baseline before each experiment."
            ),
        },
    },
    "Cybersecurity Analyst": {
        "fresher": {
            "Explain the CIA triad — Confidentiality, Integrity, and Availability.": (
                "Confidentiality ensures information is accessible only to authorised parties — implemented through encryption access controls and authentication. "
                "Integrity ensures data is accurate and has not been tampered with — implemented through hashing digital signatures and audit logs. "
                "Availability ensures systems and data are accessible when needed — implemented through redundancy failover and DDoS protection. "
                "Security decisions involve balancing all three: tightening confidentiality may impact availability. The CIA triad is the foundation for all security policy."
            ),
            "What are the most common types of cyberattacks? Name and explain three.": (
                "Phishing: fraudulent emails or websites that trick users into revealing credentials or installing malware. The most common attack vector. "
                "Ransomware: malware that encrypts victim files and demands payment for the decryption key. Delivered via phishing or unpatched vulnerabilities. "
                "SQL Injection: an attacker inserts malicious SQL into an input field to manipulate the database query — exposing or deleting data. Prevented by parameterised queries. "
                "Others include man-in-the-middle attacks DDoS denial-of-service XSS and zero-day exploits."
            ),
            "What is the difference between symmetric and asymmetric encryption?": (
                "Symmetric encryption uses the same key for both encryption and decryption — fast and suitable for bulk data. Examples: AES. "
                "Key distribution is the challenge: securely sharing the key with the recipient. "
                "Asymmetric encryption uses a public-private key pair: the sender encrypts with the recipient's public key and only the recipient's private key can decrypt it. "
                "Slower but solves key distribution. Examples: RSA ECC. "
                "In practice both are combined: asymmetric encryption exchanges a symmetric session key which then encrypts the data — as in TLS."
            ),
            "What is a firewall and how does it differ from an IDS/IPS?": (
                "A firewall is a network security device that filters traffic based on predefined rules — allowing or blocking packets based on IP ports and protocols. "
                "Stateful firewalls track connection state for more intelligent filtering. Next-gen firewalls add application awareness and SSL inspection. "
                "An IDS Intrusion Detection System monitors traffic and generates alerts on suspicious patterns but does not block traffic. "
                "An IPS Intrusion Prevention System is inline and can block malicious traffic automatically. "
                "Firewalls enforce access control policies; IDS/IPS detect and respond to active threats."
            ),
            "What is vulnerability scanning and how does it differ from penetration testing?": (
                "Vulnerability scanning is an automated process using tools like Nessus OpenVAS or Qualys to identify known vulnerabilities misconfigurations and missing patches across systems. "
                "It is non-intrusive wide in scope and run regularly. "
                "Penetration testing is a manual or semi-manual adversarial assessment where a tester actively attempts to exploit vulnerabilities to evaluate real-world risk. "
                "It is deeper more creative and scoped to specific systems or applications. "
                "Vulnerability scanning tells you what might be vulnerable; penetration testing proves what is actually exploitable."
            ),
        },
        "mid": {
            "Walk me through the incident response lifecycle (PICERL). How do you contain a ransomware incident?": (
                "PICERL: Preparation Identification Containment Eradication Recovery Lessons Learned. "
                "Ransomware response: Identification — alerts from EDR or SIEM on mass file encryption or C2 beacons. "
                "Containment — immediately isolate affected hosts from the network segment network access to stop lateral spread. "
                "Disable affected accounts. Eradication — image clean systems from known-good backups. "
                "Identify patient zero and the attack vector. Recovery — restore from offline or immutable backups test before reconnecting. "
                "Lessons learned — patch the initial vector improve segmentation and test backups regularly."
            ),
            "How do you design and implement a SIEM solution for a mid-size organization?": (
                "SIEM collects normalises and correlates log data from across the environment to detect threats. "
                "Log sources: firewalls IDS EDR Active Directory DNS cloud CloudTrail authentication systems and web servers. "
                "Choose a platform: Splunk Elastic SIEM or Microsoft Sentinel. "
                "Build detection rules mapped to MITRE ATT&CK techniques — start with high-confidence rules for credential stuffing brute force and lateral movement. "
                "Tune rules to reduce false positives. Build a tiered alert process: Tier 1 analysts triage Tier 2 investigate. "
                "Define retention policy for compliance. Test detection coverage with purple team exercises."
            ),
            "Describe a penetration testing methodology for a web application. What tools do you use?": (
                "Follow OWASP Testing Guide or PTES methodology. Phases: Reconnaissance footprinting the target's attack surface. "
                "Scanning with Burp Suite OWASP ZAP and Nikto to enumerate endpoints parameters and find misconfigurations. "
                "Exploitation: test OWASP Top 10 SQLi XSS IDOR broken auth SSRF. "
                "Post-exploitation: escalate privileges access sensitive data. "
                "Reporting: document findings with CVSS severity scores proof-of-concept screenshots and remediation recommendations. "
                "Always operate within defined scope and rules of engagement."
            ),
            "How do you design a network segmentation strategy to limit lateral movement by an attacker?": (
                "Divide the network into security zones based on sensitivity: internet-facing DMZ internal workstations servers and critical infrastructure. "
                "Use VLANs and firewall rules enforcing least-privilege communication between zones. "
                "Apply microsegmentation at the workload level with host-based firewalls or a software-defined network to restrict east-west traffic. "
                "Servers should not be able to initiate connections to workstations. "
                "Use jump hosts or PAM solutions for administrative access to sensitive segments. "
                "Validate segmentation with regular purple team exercises and network flow analysis."
            ),
            "Explain PAM (Privileged Access Management) and how it reduces the risk of insider threats.": (
                "PAM controls monitors and audits privileged account access — root admin service accounts and break-glass accounts. "
                "Features: vault credentials so admins never see plaintext passwords use just-in-time access to grant temporary elevated access record privileged sessions for forensic review. "
                "Reduces insider risk by eliminating standing privileges — admins request access only when needed with manager approval. "
                "Session recordings detect anomalous behaviour. Tools: CyberArk BeyondTrust HashiCorp Vault. "
                "Combine with MFA and separation of duties for a defence-in-depth privileged access model."
            ),
        },
        "senior": {
            "Design a zero-trust architecture for a remote-first company with 1000+ employees.": (
                "Zero trust assumes no implicit trust based on network location — every request is authenticated authorised and validated. "
                "Identity layer: SSO with MFA enforced via Okta or Azure AD. Conditional access policies deny access from unmanaged devices or anomalous locations. "
                "Device trust: enrol devices in MDM Intune or Jamf enforce compliance checks before granting access. "
                "Application access: deploy a ZTNA solution like Cloudflare Access or Zscaler replacing the corporate VPN — users access specific apps not the network. "
                "Microsegmentation for internal workloads. Continuous monitoring and UEBA to detect anomalous user behaviour post-authentication."
            ),
            "How do you build a threat hunting programme from scratch? What data sources and hypotheses do you start with?": (
                "Threat hunting is proactive analyst-led searching for threats that have evaded automated detection. "
                "Start with data sources: EDR telemetry network flow logs Active Directory logs DNS and proxy logs. "
                "Build hypotheses from threat intelligence: if APT group X is targeting our sector they would use T1059 scripting and T1027 obfuscation per MITRE ATT&CK. "
                "Hunt by querying the data for indicators of those techniques. Document findings as detections. "
                "Mature the programme by integrating CTI feeds building a hunting playbook and tracking dwell time reduction as the KPI. "
                "Run quarterly hunts aligned to the threat landscape."
            ),
            "How do you design and staff a Security Operations Center (SOC) for 24/7 coverage?": (
                "Define the SOC mission: detect respond and recover from security incidents within defined SLAs. "
                "Staffing model: three shifts for 24/7 coverage with Tier 1 analysts for alert triage Tier 2 for investigation and Tier 3 for threat hunting and engineering. "
                "Technology stack: SIEM SOAR EDR threat intelligence platform and ticketing system. "
                "SOAR automation handles repetitive Tier 1 tasks like IOC enrichment and phishing triage freeing analysts for complex cases. "
                "Define SLAs: mean time to detect under 1 hour mean time to respond under 4 hours. "
                "Build a metrics dashboard and run quarterly red team exercises to validate detection coverage."
            ),
            "Explain software supply chain security risks and how you mitigate them (SBOM, signing, SLSA framework).": (
                "Supply chain attacks compromise build tools dependencies or CI/CD pipelines to inject malicious code — SolarWinds and XZ Utils are examples. "
                "SBOM Software Bill of Materials: generate a machine-readable inventory of all components and transitive dependencies for every build. Use it for vulnerability tracking. "
                "Artifact signing: sign container images and binaries with Sigstore Cosign and enforce signature verification at deployment. "
                "SLSA Supply chain Levels for Software Artifacts: a framework of four levels of build integrity from documented to hermetically sealed reproducible builds. "
                "Also: pin dependency versions lockfile integrity checks scan dependencies in CI and vet third-party packages before adoption."
            ),
            "How do you build a security governance and compliance framework for a company targeting SOC 2 and ISO 27001?": (
                "SOC 2 covers security availability processing integrity confidentiality and privacy based on Trust Services Criteria. "
                "ISO 27001 is a comprehensive ISMS standard requiring risk assessment policy documentation and continuous improvement. "
                "Steps: define scope and risk appetite conduct a gap analysis map controls to both frameworks using a common control framework to avoid duplication. "
                "Implement required policies: access control incident response change management and vendor management. "
                "Use a GRC tool like Vanta Drata or Tugboat Logic to automate evidence collection. "
                "Engage a qualified auditor for SOC 2 Type II. Conduct annual internal audits for ISO 27001 recertification."
            ),
        },
    },
    "Data Engineer": {
        "fresher": {
            "What is ETL and how does it differ from ELT?": (
                "ETL Extract Transform Load extracts data from sources transforms it to the desired format and loads it into a target data warehouse. "
                "Transformation happens before loading often on a dedicated ETL server. Used when the target system has limited compute. "
                "ELT Extract Load Transform loads raw data into the target first then transforms it using the target system's compute power. "
                "Modern cloud data warehouses like Snowflake BigQuery and Redshift are powerful enough to handle transformation at scale making ELT the preferred modern pattern. "
                "ELT enables raw data storage and flexible late-binding transformations."
            ),
            "What is the difference between a data warehouse and a data lake?": (
                "A data warehouse stores structured processed data optimised for analytics queries. Schema-on-write — structure is defined before loading. "
                "Supports fast SQL queries from BI tools. Examples: Snowflake BigQuery Redshift. "
                "A data lake stores raw data in any format structured semi-structured or unstructured at low cost in object storage like S3. "
                "Schema-on-read — structure is applied at query time. Suited for ML workloads and exploratory analysis. "
                "A data lakehouse combines both: raw storage with ACID table support and direct SQL analytics via Delta Lake Iceberg or Hudi."
            ),
            "What is Apache Spark and what type of problems does it solve?": (
                "Apache Spark is a distributed in-memory data processing engine designed for large-scale batch and streaming analytics. "
                "It processes data across a cluster using a DAG directed acyclic graph of transformations that are optimised before execution. "
                "Solves problems where single-machine processing is insufficient: terabyte-scale ETL ML feature engineering large joins and aggregations. "
                "Spark APIs: DataFrame API SQL Streaming and MLlib. "
                "Compared to MapReduce Spark is 10-100x faster because it caches intermediate results in memory rather than writing to disk between stages."
            ),
            "Explain the difference between batch processing and stream processing.": (
                "Batch processing collects and processes data in large discrete chunks at scheduled intervals — nightly ETL jobs are typical. "
                "High throughput but high latency — results are available only after the batch completes. "
                "Stream processing handles data continuously as it arrives with low latency — seconds or milliseconds. "
                "Used for real-time dashboards fraud detection and alerting. Tools: Apache Kafka Apache Flink and Spark Streaming. "
                "Lambda architecture combines both: a batch layer for historical accuracy and a speed layer for low-latency approximations."
            ),
            "What is a data pipeline and what are its key components?": (
                "A data pipeline is an automated series of steps that moves and transforms data from source systems to destination systems. "
                "Key components: ingestion layer collects data from APIs databases files or streams. "
                "Processing layer applies transformations filtering aggregations and joins. "
                "Storage layer persists processed data in a data warehouse or lake. "
                "Orchestration layer schedules monitors and retries pipeline steps — tools include Apache Airflow Prefect and Dagster. "
                "Observability: data quality checks alerting on failures and lineage tracking."
            ),
        },
        "mid": {
            "Design a data ingestion pipeline to handle 1TB of daily event data with SLA requirements.": (
                "Use a message queue like Kafka or Kinesis to decouple producers from the pipeline and buffer spikes. "
                "Partition events by source type to allow parallel processing. "
                "Consumers write raw events to S3 or GCS in Parquet format partitioned by date and hour. "
                "Run hourly Spark or dbt jobs to clean deduplicate and aggregate data into the warehouse. "
                "For sub-hour SLA use micro-batch processing with Spark Structured Streaming or Flink. "
                "Add data quality checks at ingestion and transformation stages alerting on schema violations null rates and row count anomalies."
            ),
            "How do you implement data quality checks and validation in a production pipeline?": (
                "Define expectations upfront: row counts null rates value ranges referential integrity and schema conformance. "
                "Use Great Expectations or dbt tests to encode and run these checks automatically. "
                "Place checks at the ingestion boundary to catch source data issues and at the transformation output before data reaches consumers. "
                "On failure quarantine the bad data to a dead-letter table and trigger alerts. "
                "Build a data quality dashboard showing metrics over time to detect gradual drift. "
                "Document data contracts with source system owners to align on schema stability and SLA expectations."
            ),
            "Compare star schema and snowflake schema for a data warehouse. When do you use each?": (
                "Star schema has a central fact table connected directly to denormalised dimension tables — fewer joins faster query performance simple to understand. "
                "Ideal for most analytical workloads and BI tool consumption. "
                "Snowflake schema normalises dimension tables into multiple related tables reducing storage redundancy but requiring more joins and adding query complexity. "
                "Use star schema for most analytical use cases where query performance and simplicity matter. "
                "Use snowflake schema when storage costs are significant or when dimension data is frequently updated and maintaining consistency in a single large table is difficult."
            ),
            "How do you choose between Apache Kafka and a traditional message queue for streaming data?": (
                "Traditional queues like RabbitMQ or SQS are designed for task distribution — messages are consumed and deleted making them great for work queues and microservice decoupling. "
                "They scale to moderate throughput but do not support replay or multiple independent consumers of the same message stream. "
                "Kafka is a distributed log — messages are retained for a configurable period allowing multiple consumer groups to read independently and enabling replay for new consumers or recovery. "
                "Choose Kafka when you need high throughput event sourcing audit logs or decoupling multiple downstream consumers. "
                "Choose a traditional queue for simple task dispatch where replay and fan-out are not needed."
            ),
            "What is a data catalog and how does it support data governance?": (
                "A data catalog is a centralised inventory of an organisation's data assets — tables columns owners lineage and documentation. "
                "It enables data discovery: analysts can search for datasets without knowing which team owns them. "
                "Supports data governance by tracking: data lineage showing where data comes from and how it transforms sensitivity classifications for PII and regulatory compliance access policies and data quality metrics. "
                "Tools: Apache Atlas DataHub Alation Collibra. "
                "A good catalog is populated automatically from ETL pipelines and schema registries not maintained manually."
            ),
        },
        "senior": {
            "Design a data mesh architecture for a large enterprise with 20+ domain teams.": (
                "Data mesh decentralises data ownership: each domain team owns its data products end-to-end rather than a central data team owning everything. "
                "Four principles: domain ownership data as a product self-serve data infrastructure and federated computational governance. "
                "Implementation: each domain publishes its data products to a shared data catalogue with defined SLAs schemas and quality metrics. "
                "A central platform team provides the self-serve infrastructure: templated pipelines query engines and governance tooling. "
                "A federated governance model enforces global standards — PII handling schema versioning and access controls — without centralising ownership. "
                "Challenges: data product quality consistency and the cultural shift to domain accountability."
            ),
            "How do you architect a real-time analytics platform handling petabyte-scale event streams?": (
                "Ingestion: Kafka cluster with enough partitions for parallelism sized for peak throughput. "
                "Stream processing: Flink or Spark Structured Streaming for stateful aggregations with exactly-once guarantees. "
                "Storage: write aggregated results to an OLAP database like ClickHouse Apache Druid or Pinot designed for real-time analytics queries. "
                "Cold path: simultaneously write raw events to a data lake partitioned by time for historical queries. "
                "Serving layer: expose metrics via an API or BI tool with sub-second query response. "
                "Operational considerations: Kafka consumer lag monitoring Flink checkpointing for recovery and data schema evolution via a schema registry."
            ),
            "How do you build a data governance framework covering lineage, quality, access control, and compliance?": (
                "Lineage: instrument ETL pipelines with OpenLineage to automatically capture upstream and downstream dataset relationships. Visualise in DataHub or Marquez. "
                "Data quality: define and enforce data contracts using Great Expectations or dbt tests at every pipeline stage. "
                "Access control: implement column-level security for PII in the warehouse using role-based access. Integrate with the IAM provider. "
                "Compliance: classify data by sensitivity tag PII fields enforce data residency via regional storage apply retention policies and enable audit logging of all data access. "
                "Governance council: cross-functional team setting standards with data stewards accountable in each domain."
            ),
            "Design the feature engineering infrastructure for an ML platform serving 50+ data scientists.": (
                "Build a feature store with an online low-latency store like Redis and an offline store in the data warehouse or lake. "
                "Features are defined as code in a shared repository reviewed and versioned. "
                "Backfill pipelines populate the offline store for training. Online serving pipelines keep the online store fresh via Kafka. "
                "This ensures training-serving skew is eliminated — models train and serve the same feature logic. "
                "Expose a Python SDK for data scientists to define features retrieve training datasets and register features to the catalogue. "
                "Tools: Feast Tecton Hopsworks or a custom build on Spark and Redis."
            ),
            "How do you approach cloud data infrastructure cost optimisation while maintaining performance SLAs?": (
                "Visibility first: tag all resources and build per-team cost dashboards using cloud billing APIs or tools like Cost Explorer. "
                "Compute: right-size Spark clusters using auto-scaling and spot or preemptible instances for batch workloads saving 60-80%. "
                "Storage: enforce lifecycle policies moving cold data to cheaper tiers. Use columnar formats like Parquet with predicate pushdown to reduce bytes scanned. "
                "Warehouse: use query cost controls in BigQuery partition pruning and materialise expensive repeated queries. "
                "Architectural: avoid data duplication across systems consolidate pipelines and eliminate unused tables identified by the data catalogue. "
                "FinOps culture: make teams accountable for their compute costs with monthly reviews."
            ),
        },
    },
    "Mobile Developer": {
        "fresher": {
            "What is the difference between native, hybrid, and cross-platform mobile development?": (
                "Native development uses platform-specific languages and tools: Swift/Objective-C for iOS and Kotlin/Java for Android. "
                "Best performance full access to platform APIs and the most polished UX but requires separate codebases. "
                "Hybrid apps wrap a web app in a native container using tools like Cordova or Ionic — write once but limited access to native APIs and lower performance. "
                "Cross-platform frameworks like React Native and Flutter share a single codebase while compiling to native widgets React Native or using a custom rendering engine Flutter. "
                "Good performance and near-native UX with code sharing making them the most popular modern approach."
            ),
            "Explain the Android activity lifecycle or iOS UIViewController lifecycle.": (
                "Android Activity lifecycle: onCreate called when the activity is first created — initialise UI and data. "
                "onStart when becoming visible onResume when interactive onPause when partially obscured. "
                "onStop when fully hidden onDestroy when being destroyed. "
                "Save state in onSaveInstanceState for config changes. "
                "iOS UIViewController: viewDidLoad for one-time setup viewWillAppear/viewDidAppear before/after visible. "
                "viewWillDisappear/viewDidDisappear before/after hidden. deinit for cleanup. "
                "Understanding lifecycle is critical for managing resources preventing memory leaks and saving UI state."
            ),
            "What is state management in React Native or Flutter and why is it important?": (
                "State is data that changes over time and drives UI updates. Managing it correctly avoids bugs like stale data UI inconsistencies and unnecessary re-renders. "
                "React Native: useState for local component state useReducer for complex local logic Redux or Zustand for global shared state. "
                "Flutter: setState for simple local state Provider InheritedWidget or Riverpod for widget tree state BLoC for complex event-driven state. "
                "Poor state management leads to prop drilling duplicated state and hard-to-debug UI bugs especially as apps grow."
            ),
            "How does push notification delivery work on iOS and Android?": (
                "Both platforms use their own push notification services: APNs Apple Push Notification service for iOS and FCM Firebase Cloud Messaging for Android. "
                "Flow: the app requests permission and registers with APNs/FCM receiving a device token. "
                "The token is sent to your backend server. "
                "To send a notification the backend calls APNs or FCM HTTP API with the device token and payload. "
                "The platform service delivers the notification to the device. "
                "Background delivery uses silent pushes or background fetch modes. Handle token refresh as tokens can change."
            ),
            "What is offline-first development and how do you implement local data storage?": (
                "Offline-first means the app is fully functional without network connectivity by storing data locally and syncing when connectivity is restored. "
                "For simple key-value storage use AsyncStorage React Native or SharedPreferences Android. "
                "For structured data use SQLite via libraries like expo-sqlite or Room on Android and Core Data on iOS. "
                "Sync strategy: queue mutations locally when offline replay against the server when back online resolving conflicts with last-write-wins or user-prompt strategies. "
                "Offline-first is critical for apps used in low-connectivity environments like field work or travel."
            ),
        },
        "mid": {
            "How do you diagnose and fix jank (frame drops) in a mobile application?": (
                "Jank occurs when frames take longer than 16ms to render missing the 60fps target. "
                "Diagnosis: use React Native Performance Monitor or Android Profiler or Xcode Instruments Time Profiler to identify dropped frames. "
                "Common causes: heavy work on the JS/UI thread large list re-renders expensive layout passes and synchronous native module calls. "
                "Fixes: move heavy computation off the main thread using web workers or native modules. "
                "Virtualize long lists with FlashList or RecyclerView. Memoize components with React.memo and useMemo. "
                "Use InteractionManager.runAfterInteractions to defer non-critical work."
            ),
            "Compare Redux, MobX, and React Query for state management in React Native.": (
                "Redux is a predictable centralised store with unidirectional data flow great for complex apps with many state interactions but verbose without Redux Toolkit. "
                "MobX uses reactive observables — state changes automatically propagate to components. Less boilerplate but harder to debug due to implicit reactivity. "
                "React Query is not a general state manager but a server state library — it handles fetching caching synchronisation and background refresh of remote data automatically. "
                "Use React Query for all server data Redux or Zustand for complex client-side state and Context for simple global config. "
                "Combining React Query with Zustand covers most app needs with minimal boilerplate."
            ),
            "What are the key mobile security best practices for storing sensitive data?": (
                "Never store sensitive data like tokens or PII in AsyncStorage or localStorage — these are unencrypted. "
                "Use the platform secure storage: iOS Keychain and Android Keystore via react-native-keychain. "
                "For offline databases use SQLCipher for encrypted SQLite. "
                "Certificate pinning prevents MITM attacks by validating the server's TLS certificate against a bundled copy. "
                "Obfuscate the JS bundle with tools like Hermes and code obfuscation to slow reverse engineering. "
                "Request only necessary permissions at the appropriate time. Implement biometric authentication for high-sensitivity operations."
            ),
            "How do you set up a CI/CD pipeline for mobile app releases to the App Store and Play Store?": (
                "Use Fastlane to automate certificate management code signing and store submission. "
                "CI triggers on PR merge: run unit and integration tests on an emulator or simulator using GitHub Actions or Bitrise. "
                "On merge to main: increment build number sign the app and distribute to TestFlight App Store Connect or Firebase App Distribution for QA. "
                "For production: manual approval gate then Fastlane deliver uploads to the App Store. "
                "Manage signing certificates and provisioning profiles in CI secrets. "
                "Use automated screenshot and metadata generation with Fastlane Snapshot and Deliver."
            ),
            "How do you implement accessibility in a mobile app for screen reader users?": (
                "Ensure all interactive elements have accessible labels using accessibilityLabel in React Native or contentDescription in Android. "
                "Set accessibilityRole to describe element type: button link header. "
                "Use accessibilityHint for non-obvious actions. Ensure focusable order follows visual reading order. "
                "Test with TalkBack on Android and VoiceOver on iOS navigating the app entirely by gesture and audio. "
                "Ensure touch targets are at least 44x44 points. Avoid colour as the sole indicator of state. "
                "Caption videos and provide text alternatives for images."
            ),
        },
        "senior": {
            "How would you architect a large-scale React Native or Flutter app used by 10M users?": (
                "Feature-based monorepo with clear module boundaries: each feature is a self-contained package with its own components state and navigation. "
                "State: React Query for server state Zustand for global client state with strict separation. "
                "Navigation: centralised typed navigation using React Navigation with deep-link support. "
                "Performance: lazy-load feature modules enable Hermes JS engine use FlashList for all lists. "
                "Backend for frontend: a BFF layer tailors API responses to mobile reducing over-fetching. "
                "OTA updates: use EAS Update or CodePush for JS-layer updates between releases to fix critical bugs without App Store review. "
                "Observability: Sentry for crash reporting with session replay and custom performance traces."
            ),
            "How do you design a mobile SDK for third-party developers — API surface, versioning, and documentation?": (
                "Define the minimum viable public API surface — expose only what third parties need keeping implementation details private. "
                "Use semantic versioning: patch for bug fixes minor for backwards-compatible features major for breaking changes. "
                "Maintain a changelog and migration guides for major versions. "
                "Design for discoverability: use descriptive method names consistent naming conventions and rich TypeScript typings. "
                "Documentation: auto-generate API reference from doc comments publish interactive examples and provide a sample app. "
                "Test via a consumer-driven contract: write tests from the perspective of how developers will use the SDK. "
                "Provide a sandbox mode for testing without backend credentials."
            ),
            "How do you build and scale a cross-platform performance testing and monitoring strategy?": (
                "Synthetic performance tests: automate flows like startup time login and home screen render using Maestro or Detox measuring frame rate and time-to-interactive. "
                "Run on real device farms using AWS Device Farm or Firebase Test Lab. "
                "Track Core App Performance metrics: app startup time frame render time JS bundle load time and memory footprint across OS versions. "
                "Real user monitoring: instrument with Firebase Performance or custom traces sending percentile metrics to your data platform. "
                "Regression alerts: set p95 thresholds on key metrics and alert if a release introduces a regression. "
                "Build a performance dashboard comparing each release."
            ),
            "How do you design an A/B testing and feature flagging system for a mobile application?": (
                "Use a remote config and experimentation platform like Firebase Remote Config LaunchDarkly or a custom system. "
                "Feature flags decouple deployment from release: ship code behind a flag then enable it remotely for a percentage of users without a new app release. "
                "A/B testing: randomly assign users to control and treatment groups on the backend ensuring sticky assignment so a user always sees the same variant. "
                "Log experiment exposure events and outcome metrics to the analytics pipeline. "
                "Use sequential testing or Bayesian analysis for early stopping. "
                "Automatically disable a variant if guardrail metrics like crash rate degrade."
            ),
            "How do you manage a 50M daily active user mobile app with a 5-person engineering team?": (
                "Ruthless prioritisation: focus on reliability performance and features that move the key metric — everything else is deferred. "
                "Automation as a force multiplier: automate CI/CD testing and release so the team spends time building not maintaining pipelines. "
                "OTA updates: push JS fixes without App Store review to recover from production issues in hours not weeks. "
                "On-call rotation: each engineer takes a week of on-call with clear runbooks for the top 10 incident types. "
                "Observability: comprehensive crash reporting and real user monitoring to find and fix issues before users report them. "
                "Vendor partnerships: use managed backend services push notifications analytics and auth to avoid building infrastructure from scratch."
            ),
        },
    },
    "QA Engineer": {
        "fresher": {
            "What is the difference between manual and automated testing?": (
                "Manual testing is performed by a human tester who executes test cases without scripts — valuable for exploratory testing usability and UX where human judgement is needed. "
                "Automated testing uses scripts and tools to execute predefined test cases programmatically — faster for regression testing and repeatable at scale. "
                "Automation requires upfront investment in writing and maintaining scripts. "
                "The two complement each other: automate repetitive regression tests and use manual testing for new features exploratory scenarios and complex UX flows."
            ),
            "Explain unit, integration, and end-to-end tests — when do you use each?": (
                "Unit tests verify individual functions or classes in isolation with all dependencies mocked — fast cheap and the foundation of the test pyramid. "
                "Run thousands of unit tests in seconds. "
                "Integration tests verify that multiple components interact correctly — testing a service with its real database or testing API contracts between services. "
                "Slower than unit tests but catch interface bugs. "
                "End-to-end tests drive the full application through a browser or device verifying complete user flows — slowest most brittle and most expensive. "
                "Use the test pyramid: many unit tests fewer integration tests few E2E tests."
            ),
            "What makes a good test case? Walk me through writing one.": (
                "A good test case has: a clear description of what is being tested a specific precondition distinct test steps an expected result and a pass/fail criterion. "
                "It tests one thing only is repeatable and independent of other tests. "
                "Example for a login form: precondition — user is registered. "
                "Steps: navigate to /login enter valid credentials click Submit. "
                "Expected result: redirected to dashboard with user's name displayed. "
                "Also write negative cases: invalid password should show an error message. Good test cases are reviewed with developers before implementation."
            ),
            "What is regression testing and why is it important?": (
                "Regression testing verifies that new code changes have not broken existing functionality. "
                "As features are added bugs fixed or code refactored previously working features can inadvertently break — this is a regression. "
                "Running a regression suite after every change catches these bugs before they reach users. "
                "Manual regression testing is slow and error-prone for large applications making test automation critical. "
                "In CI/CD pipelines automated regression suites run on every pull request providing a safety net for continuous deployment."
            ),
            "Describe the bug life cycle from discovery to closure.": (
                "New: tester discovers and logs the bug with reproduction steps environment and severity. "
                "Assigned: the bug is assigned to a developer. "
                "In Progress: developer investigates and fixes. "
                "Fixed: developer marks it fixed and it goes back to the tester. "
                "Retest: tester verifies the fix in the correct environment. "
                "Closed: fix confirmed — bug closed. "
                "If the fix is insufficient the bug is Reopened. "
                "Won't Fix or Deferred are valid resolutions for low-priority bugs. Clear reproduction steps and environment details accelerate resolution."
            ),
        },
        "mid": {
            "Design a test automation framework for a REST API using a language of your choice.": (
                "Use Python with pytest and the requests library. "
                "Structure: tests directory with feature-based modules fixtures.py for shared setup and teardown a config module for environment-specific base URLs and auth. "
                "Base test class handles authentication and provides a configured requests Session. "
                "Test structure: Arrange set up test data Act call the API Assert verify status code response schema and data. "
                "Use parameterize for data-driven tests covering edge cases. "
                "Schema validation with pydantic or jsonschema. "
                "Generate HTML reports with pytest-html run in CI and alert on failures. "
                "Tag tests with markers for smoke regression and performance to run subsets in CI."
            ),
            "How do you approach performance and load testing for a web application?": (
                "Define performance SLAs first: p95 response time under 200ms error rate below 0.1% at peak load. "
                "Use k6 Locust or JMeter to simulate user load patterns. "
                "Start with a baseline single-user test to establish normal latency. "
                "Ramp up to expected peak load then to 2x peak to find the breaking point. "
                "Identify bottlenecks: database query times CPU saturation memory leaks and external API dependencies. "
                "Fix the bottleneck retest and repeat. "
                "Run soak tests at sustained load to detect memory leaks. "
                "Integrate lightweight smoke performance tests in CI to catch regressions early."
            ),
            "What is your strategy for deciding which tests to automate and which to keep manual?": (
                "Automate tests that: run frequently are stable with deterministic outcomes are tedious to execute manually cover regression scenarios and have high business risk. "
                "Keep manual: exploratory testing one-time tests scenarios that change frequently and usability or visual design validation. "
                "Use the ROI calculation: automation cost versus the time saved over many runs. "
                "Start automating the happy path smoke tests and the most common regression scenarios first. "
                "Avoid automating flaky tests — fix or remove them as they erode trust in the suite. "
                "Review the automation strategy quarterly."
            ),
            "How do you measure test coverage and use it to prioritise testing effort?": (
                "Code coverage measures what percentage of source code lines branches or conditions are executed by the test suite — use Istanbul NYC for JS or coverage.py for Python. "
                "High coverage does not equal quality — a test can execute a line without asserting anything meaningful. "
                "Use risk-based coverage prioritisation: identify business-critical paths high-complexity code and recently changed code and ensure these have thorough test coverage. "
                "Track mutation score with tools like Stryker to measure how well tests detect code changes. "
                "Combine coverage metrics with defect density data to focus effort on the highest-risk areas."
            ),
            "How do you triage and prioritise defects when you have 50+ open bugs before a release?": (
                "Classify by severity and priority separately. Severity: Critical blocker that prevents a core workflow High major feature broken Medium degraded experience Low cosmetic. "
                "Priority: how urgently it needs fixing relative to the release. "
                "Triage session with PM engineering and QA leads: must-fix before release nice-to-have and deferred to next release. "
                "Focus on blockers and any security or data-loss bugs first. "
                "Group related bugs for batch fixing efficiency. "
                "Track defect trends — a spike in new bugs after a refactor indicates a systemic issue requiring deeper investigation."
            ),
        },
        "senior": {
            "How do you embed quality into a CI/CD pipeline — from unit tests to production monitoring?": (
                "Shift left: unit tests and linting in pre-commit hooks. Integration tests and SAST on every pull request. "
                "PR gates: tests must pass code coverage must not drop and all critical paths must be green before merge. "
                "Post-merge: run E2E smoke tests against the staging environment. "
                "Before production: run a final smoke suite against the canary deployment. "
                "Production: synthetic monitoring runs critical user flows every minute alerting on failures. "
                "Real user monitoring and error tracking Sentry capture production issues. "
                "Build a quality metrics dashboard tracking defect escape rate test execution time and flakiness rate."
            ),
            "How do you build a shift-left testing culture in an organisation that historically tested late?": (
                "Education: run workshops on the cost of late defect discovery and the benefits of TDD and early testing. "
                "Embed QA engineers in development squads rather than a separate QA department to create shared ownership of quality. "
                "Three Amigos sessions: developer QA and PM discuss requirements and define acceptance criteria together before development starts. "
                "Definition of Done includes unit test coverage integration tests and a QA sign-off. "
                "Measure and share defect escape rate by team to create accountability. "
                "Celebrate quality wins publicly. Leadership must visibly prioritise quality over speed."
            ),
            "Describe a risk-based testing strategy for a major product release under tight deadlines.": (
                "Identify the highest-risk areas: new features recently changed code legacy areas with high defect history and business-critical flows like payments and auth. "
                "Allocate testing effort proportionally to risk not by feature size. "
                "Prioritise: run full regression on critical paths reduced regression on medium-risk areas and minimal testing on unchanged low-risk areas. "
                "Use exploratory testing sessions targeted at high-risk new features. "
                "Define explicit quality exit criteria: no open critical bugs p95 smoke tests passing and performance SLAs met. "
                "Communicate scope and risk to stakeholders transparently."
            ),
            "How do you design and scale a test infrastructure for 100+ microservices?": (
                "Shared test infrastructure: a Kubernetes-based test environment per team provisioned on demand using Helm charts. "
                "Contract testing with Pact to verify API compatibility between services without full integration environments. "
                "Each service has its own CI pipeline with unit and integration tests. "
                "E2E tests live in a shared repository orchestrated by a central pipeline that deploys all services and runs the suite. "
                "Test data management: a seed service creates consistent test data across environments. "
                "Observability in test runs: distributed tracing in the test environment to diagnose flaky E2E failures. "
                "Flakiness tracking: automatically quarantine and alert on tests with failure rates above a threshold."
            ),
            "How do you hire, grow, and lead a QA team across multiple product squads?": (
                "Hiring: look for engineers who think about systems failure modes and users not just test execution. "
                "Mix of strong automation engineers and exploratory testing specialists. "
                "Structure: embedded QA engineers in each squad with a QA guild for cross-team standards tooling and knowledge sharing. "
                "Growth: create clear career ladders from QA Engineer to Senior to Staff. Invest in automation and performance engineering skills. "
                "Leadership: define a quality vision and OKRs for the team. "
                "Run monthly quality retrospectives across squads. "
                "Advocate for quality at the roadmap level ensuring testing time is factored into sprint capacity."
            ),
        },
    },
    "Mechanical Engineer": {
        "fresher": {
            "Explain the difference between stress and strain, and describe Hooke's Law.": (
                "Stress is the internal force per unit area within a material caused by an external load measured in Pascals. "
                "Strain is the dimensionless deformation per unit length resulting from that stress. "
                "Hooke's Law states that within the elastic limit stress is directly proportional to strain: stress equals Young's modulus times strain. "
                "Young's modulus E is a material property measuring stiffness. "
                "Beyond the elastic limit permanent plastic deformation occurs and Hooke's Law no longer applies."
            ),
            "What are Newton's three laws of motion and how do they apply to engineering problems?": (
                "First law inertia: a body remains at rest or in uniform motion unless acted upon by a net external force. "
                "Applied in static equilibrium analysis where a structure at rest has zero net force and zero net moment. "
                "Second law: force equals mass times acceleration used to size motors actuators and determine dynamic loads on moving components. "
                "Third law: every action has an equal and opposite reaction applied in analysing reaction forces at supports and connections in structures and mechanisms."
            ),
            "What is the first and second law of thermodynamics?": (
                "First law conservation of energy: energy cannot be created or destroyed only converted from one form to another. "
                "In engineering heat added to a system equals the change in internal energy plus work done by the system. "
                "Second law: in any energy conversion some energy is lost as heat to the surroundings and entropy of an isolated system always increases. "
                "No heat engine can be 100% efficient. "
                "These laws govern the design of engines compressors refrigeration systems and heat exchangers."
            ),
            "How do you draw and interpret a free body diagram?": (
                "A free body diagram FBD isolates a body from its environment and shows all external forces and moments acting on it. "
                "Steps: identify the body of interest draw it in isolation then add all applied loads weight normal forces friction and reaction forces at supports. "
                "Apply Newton's second law: sum of forces equals zero for static equilibrium and sum of moments about any point equals zero. "
                "FBDs are the foundation of structural stress analysis mechanism design and dynamics problems."
            ),
            "What is GD&T (Geometric Dimensioning and Tolerancing) and why is it used?": (
                "GD&T is a symbolic language on engineering drawings that defines allowable variation in form orientation location and runout of part features. "
                "Unlike simple plus/minus tolerances GD&T communicates functional requirements of the part defining design intent precisely. "
                "It enables interchangeability: any part within tolerance will assemble and function correctly. "
                "Common symbols: flatness cylindricity true position concentricity and perpendicularity. "
                "Datum references establish the coordinate system from which measurements are made."
            ),
        },
        "mid": {
            "How do you set up a Finite Element Analysis (FEA) simulation? What are the key inputs and outputs?": (
                "Inputs: CAD geometry material properties boundary conditions loads and mesh parameters. "
                "Simplify geometry by removing fillets and small features that add mesh complexity without affecting results. "
                "Mesh the model with finer mesh in stress concentration regions and coarser elsewhere. "
                "Apply constraints to prevent rigid body motion then apply loads such as pressure forces and temperatures. "
                "Outputs: stress distribution strain energy displacement and factor of safety. "
                "Validate by comparing against analytical solutions or physical tests and iterate mesh until results converge."
            ),
            "What is Design for Manufacturability (DFM) and how does it influence your design decisions?": (
                "DFM is the practice of designing parts and assemblies to simplify manufacturing reduce cost and improve quality. "
                "Principles: minimise part count combine functions into single parts and use standard off-the-shelf components. "
                "Design for the manufacturing process by allowing sufficient draft angles for injection moulding and specifying machinable features with standard tooling. "
                "Avoid tight tolerances where not functionally required as tighter tolerances exponentially increase cost. "
                "Early collaboration with manufacturing engineers catches DFM issues before tooling is committed saving significant rework cost."
            ),
            "Walk me through conducting a Failure Mode and Effects Analysis (FMEA).": (
                "FMEA is a structured risk assessment that identifies potential failure modes their effects and causes before they occur. "
                "Steps: list every component or process step then for each identify potential failure modes and the effect on the system and end user. "
                "Assign Severity Occurrence and Detection scores from 1 to 10. "
                "Calculate Risk Priority Number RPN which equals Severity times Occurrence times Detection. "
                "Prioritise actions to reduce RPN for the highest-risk items by improving design or adding detection controls. "
                "Reassess RPN after mitigation actions are implemented."
            ),
            "How do you approach thermal management in a mechanical system with high heat generation?": (
                "Identify the heat sources and their power dissipation rates then calculate thermal resistance of the conduction path from source to ambient. "
                "Design for the most efficient heat removal path: conduction through materials convection to a heat sink or liquid cooling for high power densities. "
                "Select heat sink geometry based on airflow whether forced air or natural convection and use thermal interface materials to reduce contact resistance. "
                "Validate with FEA thermal simulation before prototype then test the physical prototype with thermocouples or infrared imaging."
            ),
            "Describe your material selection process for a structural component exposed to fatigue loading.": (
                "Define the loading profile: stress amplitude mean stress frequency and total cycle count. "
                "Use an S-N curve for the candidate material to determine endurance limit or fatigue strength at the required cycle count. "
                "Apply the Goodman criterion to account for mean stress effects and calculate factor of safety as endurance limit divided by calculated stress amplitude. "
                "Consider secondary factors: weight corrosion environment operating temperature weldability and cost. "
                "Use Ashby material selection charts to visualise property tradeoffs across material families."
            ),
        },
        "senior": {
            "How do you apply a systems engineering approach to a complex multi-disciplinary mechanism?": (
                "Start with requirements capture translating customer needs into measurable engineering requirements using a requirements management tool. "
                "Develop a functional architecture decomposing top-level functions into sub-functions across mechanical electrical and software disciplines. "
                "Define interfaces between subsystems with an interface control document ICD and allocate requirements to each subsystem. "
                "Verify at each level of integration from component to assembly to system. "
                "Use a design structure matrix to manage interdependencies and conduct design reviews at key milestones with cross-disciplinary review boards."
            ),
            "What is your approach to Product Lifecycle Management (PLM) and configuration control in a large programme?": (
                "PLM manages the entire lifecycle of a product from concept through design manufacturing and service. "
                "Configuration control ensures that all changes to design documents drawings and software are tracked approved and communicated. "
                "Every drawing and model has a revision history and a change request process: engineering change request ECR followed by engineering change order ECO. "
                "Use a PLM system like Teamcenter or Windchill to manage BOMs drawings and change history. "
                "Establish a configuration control board CCB with cross-functional representation to review and approve significant changes."
            ),
            "How do you design for reliability and maintainability in a product with a 20-year service life?": (
                "Reliability: establish reliability requirements MTTF or MTBF at system and component level. "
                "Use FMEA and fault tree analysis to identify and mitigate critical failure modes. "
                "Derate components by operating at 50-70% of rated stress to extend life and select materials with proven long-term durability for the environment. "
                "Validate with accelerated life testing HALT and HASS. "
                "Maintainability: design for accessible replaceable wear parts define maintenance intervals and provide clear service documentation. "
                "Use modular architecture to allow subsystem replacement without full teardown."
            ),
            "How do you manage the transition from R&D prototype to high-volume manufacturing?": (
                "Transition through design stages: engineering prototype EVT design verification DVT and production validation PVT. "
                "At each stage tighten design tolerances qualify manufacturing processes and perform statistical process control SPC studies. "
                "Run design-for-manufacturing reviews with the production team and contract manufacturer at each gate. "
                "Conduct first article inspection FAI on the first production parts to verify the manufacturing process meets design intent. "
                "Set up incoming quality control for critical components and build a production FMEA for the manufacturing process alongside the design FMEA."
            ),
            "How do you lead and align a cross-functional engineering team across mechanical, electrical, and software disciplines?": (
                "Establish a shared programme timeline with integrated milestones that account for interdependencies between disciplines. "
                "Hold weekly cross-disciplinary syncs to surface interface risks early. "
                "Use an interface control document to freeze cross-discipline interfaces as early as possible allowing parallel development. "
                "Create a shared risk register visible to all disciplines reviewed bi-weekly. "
                "Clarify decision rights: each discipline lead owns their domain but interface decisions require joint sign-off. "
                "Celebrate cross-team milestones and attribute credit across disciplines to build a collaboration culture."
            ),
        },
    },
    "Electrical Engineer": {
        "fresher": {
            "State Ohm's law and Kirchhoff's current and voltage laws, with an example.": (
                "Ohm's law: voltage equals current times resistance. A 12V battery across a 4 ohm resistor draws 3 amperes. "
                "Kirchhoff's Current Law KCL: the sum of currents entering a node equals the sum leaving — conservation of charge. "
                "Kirchhoff's Voltage Law KVL: the sum of voltages around any closed loop equals zero — conservation of energy. "
                "These laws are the foundation for analysing any linear circuit enabling calculation of unknown voltages and currents."
            ),
            "What is the difference between AC and DC circuits?": (
                "DC Direct Current flows in one direction at a constant voltage — batteries solar cells and most electronic circuits use DC. "
                "AC Alternating Current periodically reverses direction as a sinusoidal waveform characterised by frequency amplitude and phase. "
                "Mains electricity is AC at 50Hz or 60Hz because it can be efficiently stepped up for long-distance transmission and stepped down using transformers. "
                "AC analysis uses impedance combining resistance inductance and capacitance. "
                "Power electronics convert between AC and DC using rectifiers and inverters."
            ),
            "Explain how a transistor works and its main operating regions.": (
                "A BJT Bipolar Junction Transistor is a three-terminal device with base collector and emitter. "
                "A small base current controls a larger collector current making it a current-controlled amplifier or switch. "
                "Operating regions: Cutoff — base-emitter junction not forward biased transistor is off. "
                "Active — base-emitter forward biased collector-base reverse biased transistor amplifies. "
                "Saturation — both junctions forward biased transistor is fully on used as a digital switch. "
                "MOSFETs are voltage-controlled and dominate digital circuits and power electronics due to their high input impedance."
            ),
            "What is signal-to-noise ratio (SNR) and why does it matter?": (
                "SNR is the ratio of desired signal power to background noise power typically expressed in decibels. "
                "Higher SNR means the signal is more clearly distinguishable from noise. "
                "In communications a high SNR enables higher data rates and fewer bit errors. "
                "In sensor systems low SNR makes it difficult to detect small signals accurately. "
                "Improved by increasing signal strength reducing noise through filtering shielding and choosing low-noise components or using differential signalling to reject common-mode noise."
            ),
            "What is a PCB and what are its key layers?": (
                "A Printed Circuit Board mechanically supports and electrically connects components using conductive copper traces etched on an insulating substrate. "
                "A standard four-layer PCB has: a top copper layer for signal routing a power plane for power distribution a ground plane for a low-impedance return path and a bottom copper layer for additional routing. "
                "The substrate is typically FR4 glass-epoxy laminate. "
                "Vias connect layers electrically and solder mask protects copper while silkscreen labels components."
            ),
        },
        "mid": {
            "How do you design a DC-DC buck converter? Explain the key component calculations.": (
                "A buck converter steps down voltage using a switch inductor and capacitor. "
                "Duty cycle D equals output voltage divided by input voltage. "
                "Inductor L is chosen to limit ripple current to 20-30% of output current. "
                "Output capacitor C is sized to meet the output voltage ripple specification. "
                "Select the switch MOSFET and diode for the required voltage current and switching speed. "
                "Simulate in LTspice before layout and design layout for minimum switch-node loop area to reduce EMI."
            ),
            "How do you perform a circuit simulation using SPICE — what can and cannot be modelled?": (
                "SPICE solves circuit equations numerically for voltage and current at each node. "
                "Perform DC operating point AC frequency response and transient time-domain simulations. "
                "Models well: linear and nonlinear device behaviour noise temperature effects and parasitic RLC. "
                "Struggles with: electromagnetic radiation EMI mechanical stress thermal gradients and package parasitics unless extracted from the layout. "
                "Use LTspice Cadence or Keysight ADS and validate SPICE models against component datasheets and physical measurements."
            ),
            "What are EMC and EMI and how do you design a PCB to minimise electromagnetic interference?": (
                "EMI Electromagnetic Interference is unwanted electromagnetic energy that disrupts circuit operation. "
                "EMC Electromagnetic Compatibility is the ability of a device to function correctly without causing or suffering from EMI. "
                "PCB design for EMI: keep high-speed traces short and route over a solid ground plane. "
                "Minimise the loop area of switching supply switch nodes which are the main source of radiated EMI. "
                "Place decoupling capacitors as close as possible to IC power pins. "
                "Add common-mode chokes on external interfaces and shielding enclosures for worst offenders."
            ),
            "How do you choose between a microcontroller and an FPGA for a given application?": (
                "Microcontrollers MCUs run sequential firmware on a fixed processor core ideal for control-oriented applications with modest real-time requirements and low cost. "
                "FPGAs Field-Programmable Gate Arrays implement custom digital logic in parallel hardware ideal for high-speed data processing custom interfaces and deterministic sub-microsecond timing. "
                "Choose FPGA when processing bandwidth exceeds MCU capabilities or when deterministic real-time below microsecond is required. "
                "FPGAs cost more require specialist HDL skills and consume more power than MCUs."
            ),
            "How do you integrate sensors (ADC, SPI, I2C) into a hardware design and validate the interface?": (
                "Read the sensor datasheet to determine interface type and electrical characteristics including supply voltage and logic levels. "
                "I2C: connect SDA and SCL with pull-up resistors and configure the correct clock speed. "
                "SPI: connect MOSI MISO SCLK and chip select and configure correct CPOL and CPHA mode. "
                "ADC: match sensor output range to ADC input range and add an anti-aliasing filter. "
                "Validation: use a logic analyser to capture and decode bus transactions confirming timing and data values. "
                "Compare readings to a calibrated reference instrument."
            ),
        },
        "senior": {
            "How do you architect the electrical system for a battery-powered IoT device targeting 10-year battery life?": (
                "Start with a power budget enumerating all operating modes active idle and sleep with their current draw and duty cycle. "
                "Target average current under 10 microamperes for a 1000mAh battery over 10 years. "
                "Use a microcontroller with deep sleep under 1 microampere and power-gate peripherals sensors and radio modules when not in use using load switches. "
                "Select a low-quiescent-current power supply and design the radio protocol for minimum on-air time using BLE or LoRaWAN. "
                "Measure actual current in all modes with a power analyser and validate against the budget across temperature extremes."
            ),
            "How do you design power electronics for a high-reliability safety-critical system (medical or aerospace)?": (
                "Apply derating: operate components at 50% of rated voltage and 70% of rated current to extend MTBF. "
                "Redundancy: use N+1 or fully redundant power supply topology with automatic failover. "
                "Use mil-spec or medical-grade components with qualified failure rate data. "
                "Conduct FMEA and fault tree analysis to identify single points of failure. "
                "Design for isolation meeting IEC 60601 for medical devices. "
                "Validate with HALT HASS environmental stress screening and EMC testing to applicable standards."
            ),
            "Describe hardware-software co-design — how do you partition functionality between hardware and firmware?": (
                "Co-design starts with defining system requirements then jointly evaluating which functions are best implemented in hardware versus firmware. "
                "Hardware excels at deterministic real-time tasks: interrupt handling PWM generation high-speed protocol interfaces and safety-critical watchdog functions. "
                "Firmware excels at flexibility: state machine logic communication stack management and configuration. "
                "Prototype to find the boundary and if firmware cannot meet timing requirements move the bottleneck to a hardware peripheral or FPGA block. "
                "Define the BSP Board Support Package as the interface enabling parallel hardware and firmware development."
            ),
            "How do you manage electrical safety, compliance (CE, UL), and EMC testing for a product entering global markets?": (
                "Identify target markets and applicable standards early: CE for Europe IEC 62368-1 for consumer electronics UL for North America and FCC Part 15 for EMC. "
                "Engage an accredited test laboratory early to understand requirements. "
                "Design for compliance with EMC-friendly PCB layout and safety isolation creepage and clearance distances per IEC 60664. "
                "Pre-compliance testing: use a conducted emissions bench and near-field probe to find problems before formal testing. "
                "Submit to an accredited lab for formal EMC and safety testing and maintain a technical file with a declaration of conformity."
            ),
            "How do you run bring-up and validation of a complex multi-board electrical system?": (
                "Plan a structured bring-up sequence: power up the simplest subsystem first and validate power rails before energising sensitive components. "
                "Use current-limited bench supplies for first power-on to catch short circuits safely. "
                "Validate power rails with an oscilloscope checking voltage levels ripple and transient behaviour. "
                "Bring up each interface incrementally starting with oscillator and JTAG then peripheral interfaces one at a time. "
                "Use boundary scan JTAG for structural testing of solder joints. "
                "Run a comprehensive functional test script exercising every IO interface and functional mode and document every anomaly in a bring-up log."
            ),
        },
    },
    "Embedded Systems Engineer": {
        "fresher": {
            "What is the difference between a microcontroller and a microprocessor?": (
                "A microprocessor MPU is a CPU on a chip requiring external RAM ROM and peripherals and is used in PCs and smartphones. "
                "A microcontroller MCU integrates CPU RAM Flash memory and peripherals like GPIO timers UART SPI I2C and ADC onto a single chip. "
                "MCUs are optimised for embedded control applications with low power small footprint and deterministic real-time behaviour. "
                "Examples include STM32 AVR PIC and ESP32. "
                "Choose an MCU for dedicated control tasks and an MPU when you need an operating system and rich connectivity."
            ),
            "What is an RTOS and why is it used in embedded systems?": (
                "A Real-Time Operating System RTOS provides task scheduling memory management inter-task communication and timing services for embedded systems. "
                "It allows multiple tasks to run concurrently with guaranteed worst-case response times critical for real-time control. "
                "Key features: preemptive scheduler task priority mutexes semaphores message queues and software timers. "
                "An RTOS is used when a system has multiple concurrent activities with different timing requirements that cannot be managed by a simple superloop. "
                "Examples include FreeRTOS Zephyr and RT-Thread."
            ),
            "Explain the difference between I2C, SPI, and UART communication protocols.": (
                "I2C is a two-wire serial protocol SDA and SCL supporting multiple masters and slaves on the same bus using 7-bit addresses suitable for low-speed sensor communication. "
                "SPI is a four-wire full-duplex protocol MOSI MISO SCLK and chip-select with no addressing overhead making it faster than I2C and suitable for displays ADCs and flash memory. "
                "UART is an asynchronous two-wire serial protocol with no clock line where devices must agree on baud rate. "
                "Simple point-to-point communication used for debug console GPS modules and Bluetooth modules."
            ),
            "What is memory-mapped I/O and how do you use it to control peripherals?": (
                "Memory-mapped I/O maps peripheral registers into the same address space as RAM so the CPU accesses them using normal memory read/write instructions. "
                "In a Cortex-M MCU peripheral registers start at address 0x40000000. "
                "To toggle a GPIO pin you write a 1 to the specific bit of the port output data register at its mapped address. "
                "In C volatile uint32_t pointers are used to access register addresses directly. "
                "The volatile keyword tells the compiler not to optimise away the access because the value can change outside program control."
            ),
            "What is a bootloader and what is its role in an embedded system?": (
                "A bootloader is a small firmware program that runs at power-on before the main application initialising hardware and loading or verifying the main application. "
                "In systems with OTA updates the bootloader validates the integrity of a new firmware image using a cryptographic signature before applying the update. "
                "It manages boot slot selection choosing the valid application partition and handling rollback if the new image fails validation. "
                "Examples include U-Boot for Linux-based embedded systems and MCUboot for bare-metal and RTOS systems."
            ),
        },
        "mid": {
            "How do you design task scheduling and prioritisation in an RTOS application?": (
                "Identify all tasks and their real-time requirements: period deadline and worst-case execution time WCET. "
                "Apply Rate Monotonic Scheduling RMS assigning priority inversely proportional to period: shorter period equals higher priority. "
                "Check schedulability using the RMS utilisation bound: total CPU utilisation must be below 69% for guaranteed scheduling. "
                "Avoid priority inversion by using priority inheritance mutexes. "
                "Separate fast ISRs from slower application tasks by deferring processing to a task via a queue. "
                "Measure real-world task runtimes with a logic analyser or RTOS trace tool."
            ),
            "Walk me through writing a peripheral driver (e.g., SPI) in bare-metal C.": (
                "Read the MCU reference manual for the SPI peripheral register map and initialisation sequence. "
                "Enable the peripheral clock in the RCC register and configure GPIO pins for SPI alternate function. "
                "Configure the SPI control register: clock polarity CPOL clock phase CPHA baud rate prescaler data frame size and master mode then enable the peripheral. "
                "Transmit: wait for the TXE transmit buffer empty flag then write to the data register. "
                "Receive: wait for the RXNE receive buffer not empty flag then read from the data register. "
                "Implement chip-select control around each transaction and use DMA for high-throughput transfers."
            ),
            "How do you debug an embedded system with no OS — what tools and techniques do you use?": (
                "Primary tool: JTAG/SWD debugger with GDB via J-Link or ST-Link allowing breakpoints single-stepping and inspection of registers memory and call stack. "
                "GPIO toggling: toggle a spare pin at key points in code and measure with oscilloscope or logic analyser for timing analysis. "
                "UART trace: send debug strings to a terminal via UART for printf-style debugging. "
                "Logic analyser: verify SPI I2C UART communication timing and data. "
                "Oscilloscope: verify power supply integrity clock signals and analogue interfaces."
            ),
            "How do you manage heap and stack memory in a resource-constrained microcontroller?": (
                "Prefer static allocation over dynamic heap allocation to avoid fragmentation and unpredictable failures at runtime. "
                "If the heap is used apply a memory pool allocator for fixed-size blocks. "
                "Size each RTOS task stack carefully using FreeRTOS uxTaskGetStackHighWaterMark high-water mark measurement. "
                "Enable stack overflow detection hooks in FreeRTOS. "
                "Use the linker script memory map to verify total RAM usage at build time. "
                "Avoid large local variables and allocate them statically instead."
            ),
            "How do you implement and validate a CAN bus or Modbus communication stack?": (
                "CAN: configure the CAN controller with correct bit timing for the bus speed define message frame IDs and data fields in a DBC file and implement TX and RX interrupt handlers with message queues. "
                "Validate with a CAN analyser like PEAK or Vector CANalyzer. "
                "Modbus RTU: implement serial framing CRC-16 calculation and function codes for reading and writing registers. "
                "Validate with a Modbus master simulator like ModbusPoll. "
                "Test error handling including bus-off recovery CAN error frames and Modbus timeout and exception response."
            ),
        },
        "senior": {
            "How do you design firmware architecture for a safety-critical embedded system (IEC 61508 / ISO 26262)?": (
                "Start with a safety requirements analysis and determine the required Safety Integrity Level SIL or Automotive Safety Integrity Level ASIL. "
                "Decompose into safety functions and non-safety functions with clear boundaries. "
                "Apply redundancy using a dual-channel architecture with cross-checking for ASIL-D or SIL-3. "
                "Use a certified RTOS with deterministic behaviour and memory protection and enforce MISRA-C with static analysis tools. "
                "Conduct formal code reviews and independent verification and validation. "
                "Maintain a safety case document tracing requirements to implementation and test evidence."
            ),
            "How do you design a robust over-the-air (OTA) firmware update system for 1M field-deployed devices?": (
                "Architecture: a dual-bank flash scheme with an active slot and a staging slot. "
                "The bootloader verifies the new image with a cryptographic signature before swapping slots. "
                "If the new firmware fails to boot past a watchdog timeout the bootloader rolls back to the previous image. "
                "Deliver delta updates to minimise bandwidth for battery-constrained devices. "
                "A backend campaign management system rolls out updates in batches monitoring error rates before wider rollout. "
                "Sign firmware images with a private key validate with the public key in ROM and encrypt images over the air."
            ),
            "How do you define the hardware-software interface (BSP/HAL) to maximise portability across MCU families?": (
                "Define a Hardware Abstraction Layer HAL as a C API that application code calls without knowing the underlying hardware such as hal_uart_write hal_gpio_set and hal_spi_transfer. "
                "Each MCU family gets its own HAL implementation while the application layer remains hardware-agnostic. "
                "Use a Board Support Package BSP that includes the HAL implementation for the specific board including pin mapping and clock configuration. "
                "Define a porting guide documenting what must be implemented per platform. "
                "Test the HAL contract with hardware-in-the-loop tests that run against any compliant implementation."
            ),
            "What techniques do you use to reduce power consumption to achieve 10-year battery life on a coin cell?": (
                "Duty cycle aggressively spending 99.9% of time in deep sleep below 1 microampere. "
                "Wake only on RTC alarm or external interrupt to perform work then return to deep sleep immediately. "
                "Minimise active time by precomputing look-up tables batching flash writes and using DMA without CPU involvement. "
                "Use dynamic voltage scaling and power-gate all unused peripherals and components using load switches controlled by GPIO. "
                "Measure average current with a power profiler across all real-world scenarios and temperature extremes."
            ),
            "How do you harden an embedded device against physical and remote security attacks?": (
                "Secure boot: validate each firmware stage with a cryptographic signature from an immutable ROM root of trust. "
                "Disable all unused interfaces UART debug headers and JTAG in production via fuse bits. "
                "Use TLS 1.3 for all cloud communication with certificate-based mutual authentication. "
                "Store private keys in a dedicated security element or MCU secure enclave never in readable flash. "
                "Enable read-out protection to prevent JTAG flash extraction and use tamper detection to zeroize keys on case breach. "
                "Apply stack canaries watchdog timers and input validation on all external data."
            ),
        },
    },
    "Robotics Engineer": {
        "fresher": {
            "What is ROS (Robot Operating System) and what problems does it solve?": (
                "ROS is an open-source middleware framework for robot software development providing communication infrastructure tools and libraries. "
                "It solves the problem of building robot systems from scratch by providing a publish-subscribe message passing system between nodes a parameter server and launch files for starting complex multi-node systems. "
                "Nodes are independent processes communicating via topics services and actions. "
                "A large library of existing packages covers navigation perception and control. "
                "ROS 2 adds real-time support DDS middleware and improved security over ROS 1."
            ),
            "Explain the difference between forward kinematics and inverse kinematics.": (
                "Forward kinematics FK calculates the position and orientation of the end-effector given the joint angles and has a unique solution. "
                "Inverse kinematics IK solves the opposite: given a desired end-effector position find the joint angles that achieve it. "
                "IK is more complex — there may be multiple solutions no solutions or a continuous family of solutions for redundant robots. "
                "IK is solved analytically for simple manipulators or numerically using Jacobian pseudoinverse or optimisation methods for complex arms."
            ),
            "What sensors are commonly used in robotics and what does each measure?": (
                "LIDAR measures distance by emitting laser pulses and measuring return time used for mapping and obstacle detection. "
                "Camera provides rich visual information: RGB for colour and depth cameras for 3D perception. "
                "IMU Inertial Measurement Unit measures linear acceleration and angular rate used for orientation estimation and odometry. "
                "Encoders measure motor shaft rotation to track position and velocity. "
                "Force-torque sensors at the wrist of a manipulator measure contact forces for compliant control."
            ),
            "What is a PID controller and how is it tuned?": (
                "A PID controller calculates an error between a desired setpoint and measured value and applies a correction combining three terms. "
                "P proportional term reduces error but may leave steady-state offset. "
                "I integral term accumulates past errors to eliminate steady-state offset. "
                "D derivative term reacts to rate of error change dampening oscillations. "
                "Tuning with Ziegler-Nichols: increase Kp until sustained oscillation then set Kp to half that value and derive Ki and Kd from the oscillation period."
            ),
            "What is path planning and name two common algorithms used for it.": (
                "Path planning finds a collision-free path from a start configuration to a goal configuration in the robot's environment. "
                "A* is a graph-based search algorithm using a heuristic to guide search efficiently — optimal when the heuristic is admissible and widely used for 2D grid navigation. "
                "RRT Rapidly-exploring Random Tree randomly samples the configuration space and incrementally builds a tree toward the goal suited for high-dimensional spaces like robot arm motion planning. "
                "RRT-Star improves on RRT by adding rewiring to asymptotically approach the optimal path."
            ),
        },
        "mid": {
            "Compare RRT, A*, and Dijkstra for robot motion planning. When do you choose each?": (
                "Dijkstra guarantees the shortest path on a weighted graph but is computationally expensive for large spaces as it explores uniformly in all directions. "
                "A* adds a heuristic guiding search toward the goal making it much faster than Dijkstra while remaining optimal — best for 2D navigation in known grid maps. "
                "RRT samples random configurations and grows a tree without a discrete graph making it efficient for high-dimensional continuous spaces like 6-DOF robot arms. "
                "RRT-Star adds rewiring to asymptotically approach the optimal path. "
                "Use A* for 2D mobile navigation and RRT/RRT-Star for high-dimensional arm motion planning."
            ),
            "How do you fuse IMU, LIDAR, and camera data for robot localisation?": (
                "IMU provides high-rate low-latency orientation and acceleration but drifts over time. "
                "LIDAR provides accurate geometry-based pose correction at lower rates. "
                "Camera provides visual features for place recognition and visual odometry. "
                "Use an Extended Kalman Filter EKF or a factor graph optimiser like GTSAM to fuse these modalities. "
                "LIDAR-IMU tight coupling with LOAM or LIO-SAM achieves centimetre-level accuracy. "
                "Visual-inertial odometry VIO works without LIDAR in feature-rich environments."
            ),
            "How do you design a control architecture for a 6-DOF robot arm performing pick-and-place?": (
                "Task planning selects the sequence of pick and place actions. "
                "Motion planning with MoveIt computes a collision-free joint trajectory to the pre-grasp approach pose. "
                "Trajectory execution: the joint trajectory controller uses PID or impedance control at each joint. "
                "Grasp planning computes the grasp pose from vision estimates of object position and orientation. "
                "Force control uses force-torque sensor feedback at the wrist for compliant insertion tasks. "
                "Safety: velocity and force limits in the controller with emergency stop on violation."
            ),
            "What simulation environments do you use for robotics development and how do you validate in sim?": (
                "Gazebo is the most widely used open-source robotics simulator integrated with ROS supporting physics sensors and plugins. "
                "Isaac Sim from Nvidia provides photorealistic rendering and GPU-accelerated physics for training and testing. "
                "PyBullet and MuJoCo are lightweight physics engines popular for reinforcement learning research. "
                "Simulation validation: run the same test cases in sim and on hardware comparing trajectory accuracy and timing. "
                "Use domain randomisation to vary physics parameters lighting and sensor noise building robustness that transfers to real hardware."
            ),
            "How do you implement object detection and pose estimation for a robotic grasping task?": (
                "Object detection: use a deep learning model like YOLOv8 trained on domain-specific data to detect objects and their 2D bounding boxes. "
                "6-DOF pose estimation: use FoundationPose or DenseFusion to estimate position and orientation from RGB-D images. "
                "Alternatively use point cloud registration ICP to align a CAD model to the observed point cloud. "
                "Calibrate the camera to the robot base using hand-eye calibration to transform detections into robot frame. "
                "Use depth filtering and outlier rejection to improve robustness."
            ),
        },
        "senior": {
            "How do you architect a full robotics software stack — from perception to planning to control?": (
                "Perception layer: sensor drivers publish raw data and processing nodes produce semantic output including obstacle maps object detections and localisation pose. "
                "World model: a centralised environment representation updated by perception. "
                "Planning layer: global planner generates high-level routes and local planner generates real-time collision-avoiding trajectories. "
                "Control layer: trajectory tracking controller translates planned paths to actuator commands with feedback. "
                "Safety layer: orthogonal to all layers monitoring watchdog timers velocity limits and emergency stop conditions. "
                "Use ROS 2 for modularity and design each layer to be replaceable for algorithm iteration."
            ),
            "How do you coordinate a fleet of autonomous mobile robots (AMRs) in a warehouse environment?": (
                "Fleet management system FMS assigns tasks to robots optimising for throughput and balancing workload. "
                "Traffic management: define traffic rules lanes and intersection priorities using a centralised reservation system to prevent deadlocks. "
                "Task allocation: use a Hungarian algorithm or auction-based assignment to minimise travel distance. "
                "Robots report state to the FMS at 1-10Hz and the FMS sends goal updates. "
                "A centralised coordinator resolves gridlocks by rerouting lower-priority robots. "
                "Test fleet behaviour at scale in simulation before physical deployment."
            ),
            "How do you design and certify a collaborative robot (cobot) safety system for human-robot interaction?": (
                "Safety standards: ISO 10218 for industrial robots and ISO/TS 15066 for collaborative operation define four collaborative modes including speed-and-separation monitoring SSM and power-and-force limiting PFL. "
                "SSM: use safety-rated LIDAR or vision to monitor human-robot separation distance and reduce speed as the human approaches. "
                "PFL: limit joint torque via compliant control so contact force stays below injury thresholds. "
                "Implement safety functions on a SIL 2 or PLd safety controller. "
                "Conduct a risk assessment per ISO 12100 and validate by measuring contact force with a calibrated force measurement device."
            ),
            "What is the sim-to-real gap and how do you minimise it when training robot policies in simulation?": (
                "The sim-to-real gap is the performance degradation when a policy trained in simulation is deployed on real hardware due to differences in physics sensor noise actuator dynamics and visual appearance. "
                "Techniques: domain randomisation varies mass friction lighting and sensor noise during training so the policy learns robustness to variation. "
                "System identification measures real robot dynamics and calibrates the simulator to match. "
                "Fine-tune the policy on a small amount of real-world data after initial sim training. "
                "Use photorealistic rendering to reduce the visual gap."
            ),
            "How do you take a robotics prototype through to commercial deployment — hardware, software, support, and scale?": (
                "Hardware: transition from prototype to a DFM-reviewed production design qualifying the supply chain certifying safety and EMC and running production validation testing. "
                "Software: build a CI/CD pipeline with automated hardware-in-the-loop HIL testing and implement OTA update capability. "
                "Deployment: run on-site installation qualification acceptance tests and operator training. "
                "Support: remote monitoring dashboard for fleet health with proactive alerting on degraded performance. "
                "Scale: modular architecture allowing rapid customisation for new customer environments. "
                "Track mean time between failures as the key reliability KPI."
            ),
        },
    },
    "Quantitative Analyst": {
        "fresher": {
            "What is quantitative finance and how does it differ from traditional finance?": (
                "Quantitative finance applies mathematical models statistical methods and computational tools to price financial instruments manage risk and develop trading strategies. "
                "Traditional finance relies on fundamental analysis qualitative judgment and market intuition. "
                "Quant finance uses stochastic calculus probability theory time series analysis and machine learning to make data-driven decisions at scale. "
                "Quants build pricing models for derivatives construct factor models for equity portfolios and develop algorithmic trading strategies. "
                "The field requires strong skills in mathematics programming and statistics."
            ),
            "What is a derivative and give examples of options, futures, and swaps.": (
                "A derivative is a financial instrument whose value is derived from an underlying asset rate or index. "
                "Option: gives the holder the right but not the obligation to buy call or sell put the underlying at a strike price before expiration. "
                "Future: a standardised exchange-traded contract obligating both parties to buy or sell at a fixed price on a future date. "
                "Swap: a private agreement to exchange cash flows such as an interest rate swap exchanging fixed for floating payments. "
                "Derivatives are used for hedging risk speculation and accessing otherwise illiquid exposures."
            ),
            "Explain the Black-Scholes model — what does it calculate and what are its assumptions?": (
                "Black-Scholes calculates the theoretical fair value of a European option using five inputs: current stock price strike price time to expiration risk-free rate and implied volatility. "
                "Assumptions: log-normal distribution of stock returns constant volatility no dividends European exercise only continuous trading no transaction costs and constant risk-free rate. "
                "In practice volatility is not constant — the volatility smile shows implied vol varies with strike and expiry motivating models like Heston and SABR."
            ),
            "What is Monte Carlo simulation and how is it used in finance?": (
                "Monte Carlo simulation generates thousands of random scenarios for asset prices using a stochastic process like geometric Brownian motion. "
                "For each path compute the payoff then average all payoffs and discount to today to get the price. "
                "Used for path-dependent options such as Asian barriers and lookbacks where closed-form solutions do not exist. "
                "Also used for portfolio risk simulation by generating correlated return scenarios and computing loss distributions. "
                "Variance reduction techniques like antithetic variates improve convergence speed."
            ),
            "What is backtesting and what are the key risks when interpreting backtest results?": (
                "Backtesting evaluates a trading strategy by simulating its performance on historical data. "
                "Key risks: overfitting optimises parameters to past data that do not generalise to live trading. "
                "Look-ahead bias uses future information not available at the time. "
                "Survivorship bias tests only on companies that survived ignoring those that went bankrupt. "
                "Transaction cost underestimation ignores slippage bid-ask spread and market impact. "
                "Use walk-forward testing and out-of-sample validation to address overfitting."
            ),
        },
        "mid": {
            "Compare delta hedging and vega hedging for an options portfolio. When do you rebalance?": (
                "Delta is the sensitivity of an option price to a move in the underlying. Delta hedging holds a short position in the underlying making the portfolio delta-neutral and removing first-order directional risk. "
                "Vega is sensitivity to implied volatility changes. Vega hedging adds offsetting options positions to neutralise volatility risk. "
                "Rebalance delta when it drifts beyond a threshold or at fixed intervals trading off hedging cost against residual risk. "
                "A gamma-aware rebalancing approach adjusts for the rate of delta change reducing hedge lag."
            ),
            "How do you construct a mean-variance efficient portfolio with real-world constraints (turnover, sector limits)?": (
                "Mean-variance optimisation maximises expected return for a given variance using the covariance matrix of asset returns. "
                "Real-world constraints: turnover limit penalises portfolio changes to reduce transaction costs. "
                "Sector exposure limits add linear inequality constraints capping allocation per sector and position limits set min and max bounds per asset. "
                "Formulate as a quadratic programming problem solved with CVXPY or a commercial solver. "
                "Use robust optimisation or Black-Litterman to reduce sensitivity to estimation error in expected returns."
            ),
            "Explain how you build a multi-factor equity model. What factors would you include and why?": (
                "A multi-factor model decomposes stock returns into exposures to common risk factors plus an idiosyncratic component. "
                "Factors: value low P/B and P/E captures return premium for cheap stocks. Momentum prior 12-month return persists due to behavioural biases. "
                "Quality high ROE and low accruals captures fundamental business strength. Low volatility and low beta captures the low-risk anomaly. "
                "Factor construction: rank all stocks compute a z-score winsorise outliers and build a long-short portfolio. "
                "Estimate a covariance matrix across factors and use it in portfolio optimisation."
            ),
            "How do you measure and manage VaR for a fixed-income portfolio? What stress scenarios do you use?": (
                "VaR for fixed income: use historical simulation or Monte Carlo simulation of yield curve movements mapping positions to risk factors including DV01 duration bucket exposures and spread durations. "
                "Apply historical yield changes over a 250-day lookback to compute the daily P&L distribution. "
                "VaR at 99% confidence is the first percentile of the loss distribution. "
                "Stress scenarios: parallel yield curve shifts steepening flattening credit spread widening and historical scenarios from the 2008 financial crisis and 2020 COVID shock."
            ),
            "Describe a pairs trading or statistical arbitrage strategy — how do you identify pairs and manage risk?": (
                "Pairs trading exploits mean reversion between two cointegrated assets. "
                "Identification: use the Engle-Granger or Johansen cointegration test to find pairs whose spread is stationary. "
                "Trading: when the spread widens beyond 2 standard deviations short the outperformer and long the underperformer then exit when spread reverts to mean. "
                "Risk management: stop-loss if spread widens beyond 3 standard deviations and monitor for regime change that breaks cointegration. "
                "Limit position size per pair and track factor exposures to ensure market neutrality."
            ),
        },
        "senior": {
            "Design the technology and data infrastructure for a high-frequency trading system with sub-millisecond latency.": (
                "Co-locate servers in the exchange data centre to minimise network latency. "
                "Use kernel bypass networking DPDK or RDMA to eliminate OS network stack overhead achieving single-digit microsecond latency. "
                "Implement the order management system in C++ with lock-free data structures and memory pools to avoid garbage collection pauses. "
                "Use FPGA for the most latency-sensitive logic: market data parsing order routing and decision logic achieving nanosecond response times. "
                "Monitoring: nanosecond-precision timestamps at each processing stage with latency regression alerts on every deployment."
            ),
            "How do you research, validate, and deploy a new alpha factor at a quantitative hedge fund?": (
                "Research: generate a hypothesis grounded in economic intuition compute the factor from raw data and measure IC information coefficient against forward returns. "
                "Backtest in a universe consistent with live investment applying realistic transaction costs and capacity constraints. "
                "Validation: out-of-sample testing on held-out data cross-sectional bootstrap and Monte Carlo significance testing. "
                "Risk review: assess factor crowding correlation to existing factors and tail risk. "
                "Paper trade for 3-6 months before live deployment and size allocation based on confidence and information ratio."
            ),
            "What is model risk and how do you build a model risk management framework for a trading desk?": (
                "Model risk is the risk of loss from incorrect model assumptions data errors or implementation bugs leading to mispricing or incorrect hedges. "
                "Framework: maintain a model inventory registering every model with its purpose assumptions and limitations. "
                "Model validation: independent review testing assumptions and stress testing outputs. "
                "Benchmark comparison: compare model prices to market prices and competitor models. "
                "Ongoing monitoring: track model P&L explain and backtesting performance alerting on significant deviations. "
                "Governance: model sign-off process before production deployment and periodic recertification."
            ),
            "How do you apply machine learning in a systematic trading strategy while controlling for overfitting?": (
                "Feature engineering: derive economically motivated features not just raw prices to give the model a reason to generalise. "
                "Overfitting controls: use walk-forward cross-validation respecting time ordering and apply regularisation or use tree-based models with depth constraints. "
                "Ensemble methods: combine multiple models trained on different feature sets and time windows to average out individual model errors. "
                "Reality check: compare out-of-sample Sharpe ratio to a null distribution from random data. If not statistically significant the model is overfit. "
                "Monitor live model decay and retrain on a schedule."
            ),
            "How do you structure and lead a quant research team — hiring, process, and performance attribution?": (
                "Hiring: hire for mathematical depth statistical rigour and coding fluency in Python and C++. "
                "Diversity of background across physicists statisticians and computer scientists improves ideation. "
                "Research process: weekly research presentations peer review of new strategies rigorous backtest standards and a hypothesis registry to avoid redundant work. "
                "Performance attribution: decompose strategy P&L into factor contributions alpha and residual and track hit rate IC and Sharpe ratio by researcher. "
                "Culture: create psychological safety for sharing negative results as failed ideas are as valuable as successes."
            ),
        },
    },
    "Bioinformatics Analyst": {
        "fresher": {
            "What is bioinformatics and how is it applied in genomics research?": (
                "Bioinformatics applies computational methods statistics and data science to analyse biological data especially large-scale molecular data like DNA RNA and protein sequences. "
                "In genomics research it is used to analyse whole-genome sequencing data to identify genetic variants associated with disease process RNA-seq data to measure gene expression and assemble de novo genomes from sequencing reads. "
                "Bioinformatics tools and pipelines transform raw sequencing data into biological insights enabling drug target discovery biomarker identification and personalised medicine."
            ),
            "Explain the central dogma of molecular biology — DNA to RNA to protein.": (
                "The central dogma describes the flow of genetic information in a cell. "
                "DNA is transcribed into messenger RNA mRNA by RNA polymerase in the cell nucleus. "
                "The mRNA is processed spliced and exported to the cytoplasm where ribosomes translate it codon by codon into a protein using transfer RNAs. "
                "The protein folds into its functional three-dimensional structure. "
                "Exceptions include reverse transcription in retroviruses which copies RNA back into DNA. "
                "Understanding this flow is fundamental to designing sequencing experiments and interpreting transcriptomic and proteomic data."
            ),
            "What is sequence alignment and why is it important? Name one algorithm used.": (
                "Sequence alignment arranges two or more DNA RNA or protein sequences to identify regions of similarity indicating functional structural or evolutionary relationships. "
                "Global alignment aligns the entire length of two sequences while local alignment finds the most similar subsequences. "
                "Smith-Waterman is a dynamic programming algorithm for optimal local alignment. "
                "Importance: identifying gene homologues annotating novel sequences detecting mutations relative to a reference genome and studying evolutionary conservation."
            ),
            "What are biological databases (NCBI, UniProt, Ensembl) and what data do they contain?": (
                "NCBI National Center for Biotechnology Information hosts GenBank nucleotide sequences RefSeq reference genomes PubMed literature and dbSNP variant databases. "
                "UniProt is the comprehensive protein sequence and functional annotation database where Swiss-Prot is manually curated and TrEMBL is computationally annotated. "
                "Ensembl provides annotated genomes for vertebrates and model organisms with gene builds regulatory features and variant effect predictions accessible via browser and REST API. "
                "These databases are the primary reference resources for mapping reads annotating variants and retrieving functional information."
            ),
            "What is a BLAST search and how do you interpret its output?": (
                "BLAST Basic Local Alignment Search Tool rapidly finds sequences in a database with significant similarity to a query sequence using a heuristic algorithm. "
                "Key output fields: E-value is the expected number of hits with this score by chance where lower E-value means more significant. "
                "Percent identity is the fraction of aligned positions that are identical. "
                "Bit score is a normalised alignment score where higher is better. "
                "An E-value below 1e-5 and identity above 40% generally indicates a homologous sequence."
            ),
        },
        "mid": {
            "Walk me through an RNA-seq analysis pipeline from raw reads to differentially expressed genes.": (
                "Quality control with FastQC and adapter trimming with Trim Galore. "
                "Alignment: align reads to the reference genome using STAR or HISAT2 outputting a BAM file. "
                "Quantification: count reads per gene using featureCounts or use pseudoalignment with Salmon for transcript-level quantification. "
                "Differential expression: import counts into R and use DESeq2 or edgeR to model count data with a negative binomial distribution normalise for library size and estimate dispersion. "
                "Output: differentially expressed genes with log2 fold-change and adjusted p-value. "
                "Downstream analysis includes pathway enrichment with clusterProfiler and GSEA."
            ),
            "How do you perform variant calling from whole-genome sequencing data? What tools do you use?": (
                "BWA-MEM2 aligns reads to the reference genome GRCh38. "
                "Mark duplicates with GATK MarkDuplicates and apply base quality score recalibration BQSR to correct systematic errors. "
                "Variant calling: GATK HaplotypeCaller calls SNPs and indels per sample producing a GVCF then joint genotyping with GenotypeGVCFs for cohorts. "
                "Variant quality score recalibration VQSR filters low-quality variants. "
                "Annotation: Ensembl VEP annotates variants with gene consequence pathogenicity predictions and population allele frequencies from gnomAD."
            ),
            "Compare AlphaFold and homology modelling for protein structure prediction — when do you use each?": (
                "Homology modelling builds a structural model based on a known template structure and is reliable when a template with high sequence identity above 30% is available using tools like MODELLER or Swiss-Model. "
                "AlphaFold 2 uses a neural network achieving near-experimental accuracy even without a template and is now the first choice for most structure prediction tasks. "
                "AlphaFold Multimer predicts protein complex structures. "
                "Homology modelling remains useful when fine-tuning for a specific template of interest or when computational resources for AlphaFold are limited."
            ),
            "How do you control for multiple testing in a genome-wide association study (GWAS)?": (
                "GWAS tests millions of SNPs for association with a phenotype and at a 0.05 threshold hundreds of thousands of false positives are expected by chance. "
                "Bonferroni correction divides the threshold by the number of tests: 5e-2 divided by 1e6 equals 5e-8 which is the standard GWAS significance threshold. "
                "Benjamini-Hochberg FDR control is less conservative controlling the expected proportion of false discoveries. "
                "LD clumping: variants in high linkage disequilibrium are not independent reducing the effective number of tests. "
                "Permutation testing computes the null distribution by permuting phenotype labels."
            ),
            "Compare Snakemake and Nextflow for bioinformatics workflow management. What are the tradeoffs?": (
                "Snakemake uses Python-based rules with pattern-matching to define inputs and outputs and is familiar to Python users with good Conda integration for environment management. "
                "Excellent for local and HPC cluster execution with straightforward scaling. "
                "Nextflow uses a Groovy-based DSL2 with a dataflow programming model and is natively designed for cloud and container-native execution with Docker Singularity and Kubernetes. "
                "A larger library of community pipelines exists via nf-core. "
                "Choose Snakemake for Python-native teams and HPC environments and Nextflow for cloud deployment and nf-core pipelines."
            ),
        },
        "senior": {
            "How do you integrate multi-omics data (genomics, transcriptomics, proteomics) to identify disease biomarkers?": (
                "Late integration analyses each omics layer independently then combines findings at the statistical or network level. "
                "Mid-level integration uses tools like MOFA Multi-Omics Factor Analysis to find latent factors explaining variation across layers. "
                "Early integration concatenates features and applies machine learning requiring careful normalisation and dimensionality reduction. "
                "Network integration builds co-expression networks and protein-protein interaction networks finding dysregulated modules using WGCNA. "
                "Validate candidate biomarkers in an independent cohort and assess clinical utility with ROC curves and cross-validation."
            ),
            "Design a cloud-based genomics data platform to store, process, and share whole-genome data for 100K patients.": (
                "Storage: petabyte-scale object storage S3 or GCS with CRAM format for aligned reads reducing storage by 60% versus BAM. "
                "Metadata catalogue: a data lake with structured metadata enabling cohort queries by phenotype ancestry and variant. "
                "Processing: scalable variant calling and QC pipelines on AWS Batch or Google Cloud Life Sciences using containerised GATK workflows managed by Nextflow or Cromwell. "
                "Access control: role-based access with audit logging and data use agreements enforced programmatically. "
                "Data sharing: GA4GH standards DRS and Beacon for federated discovery without raw data transfer. "
                "Compliance: HIPAA GDPR data residency controls and encryption at rest and in transit."
            ),
            "How do you apply machine learning for drug target identification and compound prioritisation?": (
                "Target identification: use differential expression and genetic association data to identify genes causally linked to disease. "
                "Train graph neural networks on protein-protein interaction networks to predict novel targets and use Mendelian randomisation for causal evidence from GWAS. "
                "Compound prioritisation: use QSAR models trained on ChEMBL activity data to predict activity for novel compounds. "
                "Graph-based molecular representations with message-passing neural networks capture chemical structure more effectively than fingerprints. "
                "Active learning iteratively queries the model for the most informative compounds to synthesise reducing wet-lab cost. "
                "Validate predictions prospectively in biological assays."
            ),
            "What regulatory and ethical considerations apply when running a clinical bioinformatics pipeline (HIPAA, GDPR, CLIA)?": (
                "HIPAA in the US protects protected health information PHI and genomic data linked to an individual is PHI requiring access controls audit trails and encryption. "
                "GDPR in Europe treats genetic data as a special category requiring explicit consent and data minimisation with patients having rights to access and erasure. "
                "CLIA regulates laboratories reporting clinical genetic test results requiring pipelines to be validated with documented sensitivity and specificity. "
                "Informed consent: patients must understand how their genomic data will be used stored and shared. "
                "Re-identification risk: apply k-anonymity and strict access controls as anonymised genomic data can be re-identified."
            ),
            "How do you build and lead a bioinformatics research platform team that serves wet-lab scientists at scale?": (
                "Product mindset: treat internal tools and pipelines as products with user research roadmaps and feedback loops from wet-lab scientists. "
                "Standardised pipelines: build validated reproducible pipelines using nf-core or Snakemake so scientists get consistent results without bioinformatics expertise. "
                "Self-service portal: a web interface where scientists submit samples select analyses and retrieve results without writing code. "
                "Documentation and training: comprehensive docs office hours and workshops so scientists can interpret results independently. "
                "Team structure: a mix of bioinformatics software engineers and computational biologists aligned to the most impactful scientific questions."
            ),
        },
    },
}


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_answer(user_answer: str, role: str, level: str, question: str) -> dict:
    """Analyze a single answer against the ideal answer for that role/level/question."""
    ideal = IDEAL_ANSWERS.get(role, {}).get(level, {}).get(question, "")

    if not user_answer.strip():
        return {
            "relevance_score": 0.0,
            "confidence_score": 50.0,
            "clarity_score": 0.0,
            "filler_count": 0,
            "word_count": 0,
            "final_score": 0.0,
            "feedback": ["Please provide a detailed answer."],
        }

    # ── Relevance via TF-IDF cosine similarity ────────────────────────────
    if ideal:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([ideal, user_answer])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        relevance_score = round(float(similarity) * 100, 2)
    else:
        relevance_score = 0.0

    # ── Confidence via VADER sentiment ───────────────────────────────────
    sentiment = sia.polarity_scores(user_answer)
    confidence_score = round((sentiment['compound'] + 1) * 50, 2)

    # ── Clarity via sentence length ──────────────────────────────────────
    sentences = [s.strip() for s in re.split(r'[.!?]', user_answer) if s.strip()]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        clarity_score = min(round((avg_len / 20) * 100, 2), 100)
    else:
        clarity_score = 0.0

    # ── Filler word count ────────────────────────────────────────────────
    lowered = user_answer.lower()
    filler_count = sum(
        len(re.findall(r'\b' + re.escape(w) + r'\b', lowered))
        for w in FILLER_WORDS
    )

    word_count = len(user_answer.split())

    # ── Final score ──────────────────────────────────────────────────────
    final_score = max(
        round(
            relevance_score * 0.50
            + confidence_score * 0.25
            + clarity_score * 0.25
            - filler_count * 2,
            2,
        ),
        0,
    )

    # ── Feedback ─────────────────────────────────────────────────────────
    feedback = []
    if relevance_score < 40:
        feedback.append("Add more technical keywords and domain-specific terminology.")
    elif relevance_score < 65:
        feedback.append("Expand with more relevant technical details and examples.")

    if confidence_score < 45:
        feedback.append("Use assertive, confident language — avoid hedging phrases.")
    elif confidence_score < 60:
        feedback.append("Be more decisive and direct in your phrasing.")

    if clarity_score < 40:
        feedback.append("Structure your answer in clearer, complete sentences.")

    if word_count < 30:
        feedback.append("Your answer is too brief — aim for at least 50 words per question.")

    if filler_count > 2:
        feedback.append(f"Reduce filler words ({filler_count} detected) to sound more professional.")

    if not feedback:
        feedback.append("Excellent — well-structured and technically relevant answer!")

    return {
        "relevance_score": relevance_score,
        "confidence_score": confidence_score,
        "clarity_score": clarity_score,
        "filler_count": filler_count,
        "word_count": word_count,
        "final_score": final_score,
        "feedback": feedback,
    }


def analyze_all_answers(answers: list, role: str, level: str, questions: list) -> dict:
    """Analyze all answers for an interview session and return aggregated results."""
    per_question = []
    for answer, question in zip(answers, questions):
        result = analyze_answer(answer, role, level, question)
        per_question.append(result)

    n = len(per_question) or 1
    return {
        "per_question": per_question,
        "avg_relevance": round(sum(r["relevance_score"] for r in per_question) / n, 2),
        "avg_confidence": round(sum(r["confidence_score"] for r in per_question) / n, 2),
        "avg_clarity": round(sum(r["clarity_score"] for r in per_question) / n, 2),
        "total_filler_count": sum(r["filler_count"] for r in per_question),
        "total_word_count": sum(r["word_count"] for r in per_question),
    }
