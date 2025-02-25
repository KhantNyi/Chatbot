import json
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# --- Part 1: Load and Embed Course Data ---
def load_course_data():
    client = chromadb.PersistentClient(path="./vector_db")
    collection_name = "grok_all"
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.create_collection(name=collection_name)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    json_file_path = "/dataset/dedup.json"
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)

    unique_data = {}
    for item in data:
        doc_id = item["id"]
        cleaned_metadata = {
            key: (", ".join(value) if isinstance(value, list) else value)
            for key, value in item["metadata"].items()
        }
        page_content = item.get("text", "").strip()
        course_key = f"{cleaned_metadata.get('course_name', doc_id)}-{cleaned_metadata.get('year', 'unknown')}"

        if not page_content:
            print(f"⚠️ Skipping {doc_id} due to missing text!")
            continue

        if course_key not in unique_data:
            unique_data[course_key] = {
                "id": doc_id,
                "page_content": page_content,
                "metadata": cleaned_metadata
            }

    texts = [item["page_content"] for item in unique_data.values()]
    ids = [item["id"] for item in unique_data.values()]
    metadatas = [item["metadata"] for item in unique_data.values()]
    embeddings = [embedding_model.encode(text).tolist() for text in texts]

    try:
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=texts
        )
        print(f"✅ Embedded {len(texts)} unique items into '{collection_name}'.")
    except Exception as e:
        print(f"❌ Error adding data to ChromaDB: {e}")
        exit(1)

    return client, collection

# --- Part 2: Load and Embed FAQ Data ---
def load_faq_data(client):
    faq_collection_name = "faqs_vectors"
    try:
        client.delete_collection(faq_collection_name)
    except ValueError:
        pass

    faq_collection = client.create_collection(name=faq_collection_name)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    faq_json_path = "Chatbot/expanded_faq_recommendations.json"
    try:
        with open(faq_json_path, "r", encoding="utf-8") as f:
            faqs = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading JSON file: {e}")
        exit(1)

    texts = [faq["question"] for faq in faqs]
    ids = [f"faq-{i+1}" for i in range(len(faqs))]
    embeddings = [embedding_model.encode(text).tolist() for text in texts]

    def format_answer(answer):
        if isinstance(answer, list):
            return ", ".join(answer)
        return answer

    metadatas = [{"question": faq["question"], "answer": format_answer(faq["answer"])} for faq in faqs]

    faq_collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Stored {len(faqs)} FAQs into '{faq_collection_name}'.")
    return faq_collection

# --- Part 3: Chatbot Logic ---
def setup_chatbot(client, course_collection, faq_collection):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(client=client, collection_name="grok_all", embedding_function=embedding_function)

    generator = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
    llm = HuggingFacePipeline(pipeline=generator)

    def preprocess_query(query):
        query_lower = query.lower().strip()
        # Aggressively strip prerequisite-related terms and extra words
        cleaned_query = query_lower.replace("prerequisites for ", "").replace("prerequisite for ", "").replace("prerequisite ", "").replace("requirements for ", "").replace("rerequisite for ", "").replace("prerequsite for ", "").replace("what's the ", "").replace("for ", "").strip()

        year = "65"  # Default to 2025 (Year 65)
        if "year 64" in cleaned_query:
            year = "64"
            cleaned_query = cleaned_query.replace("year 64", "").strip()
        elif "year 65" in cleaned_query:
            year = "65"
            cleaned_query = cleaned_query.replace("year 65", "").strip()

        cleaned_query = cleaned_query.replace("cs", "").replace("it", "").strip()
        cleaned_query = cleaned_query.replace("cybesecurity", "cybersecurity").replace("enterprsie", "enterprise").replace("quantitive", "quantitative").replace("digal", "digital").replace("prerequse", "").replace("biometri", "biometrics").replace("sciece", "science").strip()

        course_name_variations = {
            "fundamental of financial accounting": "ACT1601 Fundamental of Financial Accounting",
            "fundamentals of financial accounting i": "BAC1602 Fundamentals of Financial Accounting I",
            "business exploration": "BBA1001 Business Exploration",
            "business law i": "BG1400 Business Law I",
            "basic mathematics and statistics": "CSX1001 Basic Mathematics and Statistics",
            "introduction to information technology": "CSX2001 Introduction to Information Technology",
            "calculus": "CSX2002 Calculus",
            "principles of statistics": "CSX2003 Principles of Statistics",
            "ui/ux design and prototyping": "CSX2004 UI/UX Design and prototyping",
            "design thinking": "CSX2005 Design Thinking",
            "mathematics and statistics for data science": "CSX2006 Mathematics and Statistics for Data Science",
            "data science": "CSX2007 Data Science",
            "mathematics foundation for computer science": "CSX2008 Mathematics Foundation for Computer Science",
            "cloud computing": "CSX2009 Cloud Computing",
            "fundamentals of computer programming": "CSX3001 Fundamentals of Computer Programming",
            "object-oriented concepts and programming": "CSX3002 Object-Oriented Concepts and Programming",
            "data structure and algorithms": "CSX3003 Data Structure and Algorithms",
            "programming languages": "CSX3004 Programming Languages",
            "computer networks": "CSX3005 Computer Networks",
            "database systems": "CSX3006 Database Systems",
            "computer architecture": "CSX3007 Computer Architecture",
            "operating systems": "CSX3008 Operating Systems",
            "algorithm design": "CSX3009 Algorithm Design",
            "senior project i": "CSX3010 Senior Project I",
            "senior project ii": "CSX3011 Senior Project II",
            "information system analysis and design": "CSX4101 Information System Analysis and Design",
            "software engineering": "CSX4102 Software Engineering",
            "requirement engineering": "CSX4103 Requirement Engineering",
            "software testing": "CSX4104 Software Testing",
            "it project management": "CSX4105 IT Project Management",
            "enterprise architecture": "CSX4106 Enterprise Architecture",
            "web application development": "CSX4107 Web Application Development",
            "ios application development": "CSX4108 iOS Application Development",
            "andriod application development": "CSX4109 Andriod Application Development",
            "backend application development": "CSX4110 Backend Application Development",
            "selected topic in requirement engineering": "CSX4181 Selected Topic in Requirement Engineering",
            "selected topic in it project management": "CSX4182 Selected Topic in IT Project Management",
            "artificial intelligence concepts": "CSX4201 Artificial Intelligence Concepts",
            "data mining": "CSX4202 Data Mining",
            "machine learning": "CSX4203 Machine Learning",
            "biometrics": "CSX4204 Biometrics",
            "big data analytics": "CSX4205 Big Data Analytics",
            "data warehousing and business intelligence": "CSX4206 Data Warehousing and Business Intelligence",
            "decision support and recommender systems": "CSX4207 Decision Support and Recommender Systems",
            "deep learning": "CSX4208 Deep Learning",
            "intelligent system development": "CSX4209 Intelligent System Development",
            "natural language processing and social interactions": "CSX4210 Natural Language Processing and Social Interactions",
            "data engineering": "CSX4211 Data Engineering",
            "data analytics": "CSX4212 Data Analytics",
            "computer vision": "CSX4213 Computer Vision",
            "network design": "CSX4301 Network Design",
            "cisco networking workshop": "CSX4302 Cisco Networking Workshop",
            "network security": "CSX4303 Network Security",
            "network management": "CSX4304 Network Management",
            "heterogeneous wireless networks": "CSX4305 Heterogeneous Wireless Networks",
            "internet of things": "CSX4306 Internet of Things",
            "business continuity planning and management": "CSX4307 Business Continuity Planning and Management",
            "business systems": "CSX4401 Business Systems",
            "sales and distribution mangement system": "CSX4402 Sales and Distribution Mangement System",
            "manufacturing management system": "CSX4403 Manufacturing Management System",
            "supply chain management system": "CSX4404 Supply Chain Management System",
            "finance and accounting information system": "CSX4405 Finance and Accounting Information System",
            "customer relationship mangement system": "CSX4406 Customer Relationship Mangement System",
            "enterprise application development": "CSX4407 Enterprise Application Development",
            "enterprise database system": "CSX4408 Enterprise Database System",
            "blockchain technology": "CSX4409 Blockchain Technology",
            "theory of computation": "CSX4501 Theory of Computation",
            "tech startup": "CSX4502 Tech Startup",
            "information systems security": "CSX4503 Information Systems Security",
            "digital marketing": "CSX4504 Digital Marketing",
            "digital transformation": "CSX4505 Digital Transformation",
            "image processing": "CSX4506 Image Processing",
            "information retrieval and search engines": "CSX4507 Information Retrieval and Search Engines",
            "quantitative research for digital business": "CSX4508 Quantitative Research for Digital Business",
            "neural network": "CSX4510 Neural Network",
            "ar/vr application development": "CSX4513 AR/VR Application Development",
            "cross-platform application development": "CSX4514 Cross-platform Application Development",
            "game design and development": "CSX4515 Game Design and Development",
            "reusability and design patterns": "CSX4516 Reusability and Design Patterns",
            "presentation and data visualization techniques": "CSX4601 Presentation and Data Visualization Techniques",
            "selected topic in entrepreneurship in technology business": "CSX4602 Selected Topic in Entrepreneurship in Technology Business",
            "selected topic in numerical analysis": "CSX4603 Selected Topic in Numerical Analysis ",
            "selected topic in information security": "CSX4605 Selected Topic in Information Security",
            "selected topic in software quality assurance": "CSX4608 Selected Topic in Software Quality Assurance",
            "selected topic in ai for business": "CSX4609 Selected Topic in AI for Business",
            "selected topic in ai prompt engineering": "CSX4610 Selected Topic in AI Prompt Engineering",
            "selected topic in quantum computing for entrepreneurs": "CSX4611 Selected Topic in Quantum Computing for Entrepreneurs",
            "selected topic in business insights and visualization": "CSX4615 Selected Topic in Business Insights and Visualization",
            "selected topic in backend application development": "CSX4616 Selected Topic in  Backend Application Development",
            "communicative english i": "ELE1001 Communicative English I",
            "communicative english ii": "ELE1002 Communicative English II",
            "academic english": "ELE2000 Academic English",
            "advanced academic english": "ELE2001 Advanced Academic English",
            "ecology and sustainability": "GE1302 Ecology and Sustainability",
            "science for sustainable future": "GE1303 Science for Sustainable Future",
            "communication in thai": "GE1403 Communication in Thai",
            "thai usage": "GE1408 Thai Usage",
            "thai language for intercultural commuinication": "GE1409 Thai Language for intercultural commuinication",
            "thai for professional communication": "GE1410 Thai for Professional Communication",
            "thai language for muliticultural": "GE1411 Thai Language for Muliticultural",
            "introductory thai usage": "GE1412 Introductory Thai Usage",
            "human heritage and globalization": "GE2102 Human Heritage and Globalization",
            "human civilizations and global citizens": "GE2110 Human Civilizations and Global Citizens",
            "ethics": "GE2202 Ethics",
            "accounting for entrepreneurs": "IBE1122 Accounting for Entrepreneurs",
            "business laws for entrepreneurs": "LAW1201 Business Laws for Entrepreneurs",
            "mathematics for business": "MA1200 Mathematics for Business",
            "introduction to business": "MGT1101 Introduction to Business"
        }

        for key, corrected in course_name_variations.items():
            if key in cleaned_query:
                return corrected, year
        return query_lower, year

    def get_course_answer(query, year, llm):
        processed_query, query_year = preprocess_query(query)
        query_embedding = embedding_model.encode(processed_query).tolist()
        results = course_collection.query(query_embeddings=[query_embedding], n_results=10)

        print(f"\n[DEBUG] Query: {processed_query} (Year: {query_year})")
        if not results["ids"][0]:
            print("[DEBUG] No results found.")
            prompt = f"I couldn't find any courses matching '{query}'. Could you clarify or ask about a specific course?"
            return llm(prompt)

        for i, (id, meta, doc, dist) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0])):
            distance = max(0, dist)
            confidence = 1 - min(distance, 1)
            print(f"[DEBUG] Result {i+1}: ID={id}, Confidence={confidence:.3f}, Course={meta.get('course_name')}, Prereq={meta.get('prerequisites')}, Year={meta.get('year')}")

        # Extract course name by removing common query terms
        query_words = processed_query.split()
        stop_words = ["prerequisite", "prerequiste", "for", "year", "65", "64", "cs"]
        query_course_name = " ".join([word for word in query_words if word not in stop_words and not word.startswith("csx")]).lower()
        matching_results = [
            (id, meta, doc, dist) for id, meta, doc, dist in zip(results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0])
            if meta.get("year") == query_year and query_course_name == meta.get("course_name", "").lower()
        ]
        if not matching_results:
            print(f"[DEBUG] No exact match for '{query_course_name}' in year {query_year}.")
            any_year_matches = [
                (id, meta, doc, dist) for id, meta, doc, dist in zip(results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0])
                if query_course_name == meta.get("course_name", "").lower()
            ]
            if any_year_matches:
                top_result = sorted(any_year_matches, key=lambda x: x[3])[0]
                course_id, metadata, document, distance = top_result
                course_name = metadata.get("course_name", "Unknown Course")
                prerequisites = metadata.get("prerequisites", "None specified")
                actual_year = metadata.get("year")
                prompt = f"The user asked: '{query}'. I couldn’t find '{course_name}' for year {query_year}, but it exists in year {actual_year} with prerequisites: {prerequisites}. Provide a natural response noting the year mismatch."
                return llm(prompt)
            else:
                print(f"[DEBUG] No match for '{query_course_name}' in any year. Falling back to top result.")
                course_id, metadata, document, distance = results["ids"][0][0], results["metadatas"][0][0], results["documents"][0][0], results["distances"][0][0]
        else:
            top_result = sorted(matching_results, key=lambda x: x[3])[0]
            course_id, metadata, document, distance = top_result

        course_name = metadata.get("course_name", "Unknown Course")
        prerequisites = metadata.get("prerequisites", "None specified")
        if prerequisites == "None specified":
            prompt = f"The user asked: '{query}'. The course '{course_name}' in year {query_year} has no prerequisites. Provide a natural, concise response stating there are no prerequisites."
        else:
            prompt = f"The user asked: '{query}'. The course '{course_name}' in year {query_year} has the following prerequisites: {prerequisites}. Provide a natural, concise response listing the prerequisites."
        return llm(prompt)

    def get_faq_answer(query, llm, threshold=0.5):
        query_embedding = embedding_model.encode(query).tolist()
        results = faq_collection.query(query_embeddings=[query_embedding], n_results=1)

        print(f"\n[DEBUG] FAQ Query: {query}")
        if not results["ids"][0]:
            print("[DEBUG] No FAQ results found.")
            return None

        distance = max(0, results["distances"][0][0])
        confidence = 1 - min(distance, 1)
        print(f"[DEBUG] FAQ Match: ID={results['ids'][0][0]}, Confidence={confidence:.3f}, Question={results['metadatas'][0][0]['question']}")
        if confidence < threshold:
            print(f"[DEBUG] FAQ confidence {confidence:.3f} below threshold {threshold}.")
            return None

        faq_answer = results["metadatas"][0][0]["answer"]
        prompt = f"The user asked: '{query}'. The relevant FAQ answer is: '{faq_answer}'. Provide a natural, concise response."
        return llm(prompt)

    def recommend_courses(query, llm):
        preference_mapping = {
            "i don't want to code": [
                "ITX3008 IT Project Management",
                "CSX2004 UI/UX Design and prototyping",
                "CSX4504 Digital Marketing",
                "ITX4509 Cybersecurity",
                "CSX4505 Digital Transformation"
            ],
            "i want to work with data": ["CSX1234 Data Science", "Big Data Analytics", "Predictive Analytics"],
            "i want to do ai": ["CSX4201 Artificial Intelligence Concepts", "Machine Learning", "Deep Learning"],
        }

        matched_courses = preference_mapping.get(query.lower(), [])
        if not matched_courses:
            return None

        courses_str = ", ".join(matched_courses)
        prompt = f"The user said: '{query}'. Recommend these courses: {courses_str}. Provide a natural, concise response."
        return llm(prompt)

    def run_chatbot():
        print("\n📚 Welcome to the University Course Advisor Chatbot! Type 'exit' to stop.\n")
        greetings = ["hi", "hello", "hey", "hi there", "hello there"]
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                print("Goodbye! 😊 If you need more help, feel free to ask anytime!")
                break

            try:
                query_lower = query.lower()
                processed_query, year = preprocess_query(query)

                if query_lower in greetings:
                    prompt = f"The user said: '{query}'. Respond casually and invite them to ask about courses or anything else."
                    response = llm(prompt)
                    print(f"\n🎓 Advisor: {response}\n")
                    continue

                if "prerequisite" not in query_lower and "requirements" not in query_lower:
                    faq_answer = get_faq_answer(query, llm)
                    if faq_answer:
                        print(f"\n🎓 Advisor: {faq_answer}\n")
                        continue

                if "prerequisite" in query_lower or "requirements" in query_lower:
                    course_answer = get_course_answer(query, year, llm)
                    if course_answer:
                        print(f"\n🎓 Advisor: {course_answer}\n")
                        continue

                recommendation_answer = recommend_courses(query, llm)
                if recommendation_answer:
                    print(f"\n🎓 Advisor: {recommendation_answer}\n")
                    continue

                prompt = f"The user asked: '{query}'. I couldn’t find a specific course, FAQ, or recommendation match. Respond naturally and suggest they ask about courses or something else."
                response = llm(prompt)
                print(f"\n🎓 Advisor: {response}\n")

            except Exception as e:
                print(f"❌ Oops! Something went wrong: {e}")

    return run_chatbot

# --- Main Execution ---
if __name__ == "__main__":
    client, course_collection = load_course_data()
    faq_collection = load_faq_data(client)
    chatbot = setup_chatbot(client, course_collection, faq_collection)
    chatbot()
