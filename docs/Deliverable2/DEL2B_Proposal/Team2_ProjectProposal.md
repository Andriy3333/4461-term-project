**Enshittification via Bot Interference**

By Group 2  
Members: Yinkan Chen, Hongji Tang, Andriy Sapeha  
Github: [https://github.com/Andriy3333/4461-term-project](https://github.com/Andriy3333/4461-term-project) 

**Section 1:**  
Our project focuses on how AI bots, large language models (LLMs), and similar entities might be contributing to social media platform degradation. While much of the blame for the decline of these platforms is placed on the companies themselves, we want to explore whether bots and AI-driven entities play a role in this process. Specifically, we’ll look at how these entities degrade content quality, increase misinformation and how their presence might be accelerating the platform’s decline. Our simulation will explore how AI bots interact with users and with each other, and how these interactions contribute to the overall quality of the platform and satisfaction of users.

**Section 2:**

#### **LLM Among Us: Generative AI Participating in Digital Discourse**

Radivojevic, K., Clark, N., & Brenner, P. (2024). *LLMs Among Us: Generative AI Participating in Digital Discourse.* Retrieved from [https://arxiv.org/html/2402.07940v1](https://arxiv.org/html/2402.07940v1)

In “LLM Among Us: Generative AI Participating in Digital Discourse,” the authors conducted an experiment testing how effectively people could identify LLMs and bots within a given context. Participants were notified beforehand that AI-generated responses were included, and the LLMs were trained using personas to make them more convincing. Conducted using university students, the study found that participants had only a 42% accuracy rate in distinguishing AI-generated text from human-written text. This statistic paints a worrying picture of how easy it will be in the future to manipulate and deceive users via AI and LLMs.

This study is highly relevant to our project as it demonstrates the increasing difficulty of detecting AI-generated content. Our research extends this by exploring AI-to-AI interactions and their role in exacerbating the spread of misinformation through self-reinforcing content loops.

#### **Large Language Models in Cybersecurity Threats, Exposure, and Mitigation**

Kucharvy, A., Plancherel, O., Mulder, V., Mermoud, A., & Lenders, V. (2024). *Large Language Models in Cybersecurity Threats, Exposure, and Mitigation.* Retrieved from [https://link.springer.com/book/10.1007/978-3-031-54827-7](https://link.springer.com/book/10.1007/978-3-031-54827-7)

Many topics within this book document details about LLMs and their impact on cybersecurity and digital discourse. Specifically, in Chapter 11, the authors discuss LLMs and their influence within social media operations. They describe how, with the advent of LLM models, it has become nearly impossible to distinguish between human- and bot-generated text. Consequently, AI bots can be leveraged for smear campaigns, misinformation amplification, and the creation of self-sustaining AI-driven echo chambers. These behaviors have immense social and psychological effects on real users.

This aligns with our project’s goal of analyzing AI-driven degradation of online discourse. We will use this insight to simulate how AI bots exploit engagement metrics to amplify misinformation, outcompete human-generated content, and erode platform integrity.

**Section 3:**  
**Entities:**  
The model of entities within our simulation are **Human Users**, **Bots** and the **Programmers** of the bots. The human users are the core users of the platform, that engage with content that can influence their daily lives or others. Bots are also capable of interacting with them on a large scale or each other to form echo chambers or misinformation campaigns. The programmers behind the bots are responsible for what the bots say or how they interact with others. Within the context of how they interact, the bots act like a virus that can taint or infect human users by spreading misinformation to them that causes genuine harm to people in real life. As such it follows the **Virus on a Network** model.

**Affordances:**  
Within social media platforms there are a couple of main features/options for users and bots to access and utilize. **Tweet/Reposting** allows for rapid distribution of information, content, and comments. Ways of interacting with these content and other people include **Liking**, **Following**, and **Commenting**. Some other features include **Hashtags**, **Mentions** and **Community Notes** that can tag specific people or leave behind some context behind some posts. Overall all of these common features found in social media platforms can contribute to **Boid Flocking** where bots or humans can lead to guiding users to specific narratives, behaviors or information.

**Algorithms:**  
	Some algorithms that exist within social media platforms include **Home Page Algorithms** that generate content based on what’s trending, what you’ve engaged with, and popularity. **Moderation Algorithms** also exist that attempt to detect, flag, or edit content that might offend the terms of services. All of these algorithms exist to engage users and follow the **Schelling Segregation Model** that can group users into existing categories of other similar people. Bots can take advantage of this and generate massive trends within the home page and/or echo chambers that keeps people trapped in one side of the spectrum. Within these echo chambers AI to AI interaction can create the illusion of acceptance within a misinformative circle.

**Section 4:**  
	For our simulation we plan to visualize the enshittification of websites, likely twitter, across time alongside documenting the advances of bots and the number of them and human users. Another factor that we must account for is the amount of posts attributed to them. As such we want to be able to visualize both the amount of bots and human posters and the amount of content they’re posting. 

In the diagram is an example of what we’re trying to visualize. The intense color indicates how active the entity is and the overlap of color shows what portion of the user base has interacted with each other. Additionally we want to see if we can identify what percentage of these echo chambers are a result of AI to AI interactions.

We will track key metrics such as:

* **Bot Growth Rate:** The increase in AI-generated content relative to human-generated content.  
* **Echo Chamber Formation:** The extent to which AI bots interact among themselves, reinforcing biased perspectives.  
* **Misinformation Amplification:** The spread and engagement levels of AI-generated false narratives.

Additionally, we will incorporate a **visualization model** where:

* Nodes represent human users, AI bots, and misinformation bots.  
* Connections illustrate content engagement, highlighting AI-driven clustering.  
* Color intensity signifies bot activity, showing how misinformation clusters evolve over time.

This simulation will help us measure the extent to which AI-driven interactions accelerate platform degradation and influence public discourse.

Citations:  
[*LLMs Among Us: Generative AI Participating in Digital Discourse* by Kristina Radivojevic, Nicholas Clark, Paul Brenner, 08 Feb, 2024](https://arxiv.org/html/2402.07940v1) (1) (New Source)

[Large Language Models in Cybersecurity Threats, Exposure and Mitigation, by Andrei Kucharvy, Octave Plancherel, Valentin Mulder, Alain Mermoud, Vincent Lenders, 2024](https://link.springer.com/book/10.1007/978-3-031-54827-7) (2) (New Source)