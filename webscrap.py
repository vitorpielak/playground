import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import time
import json
from urllib.parse import quote_plus
import matplotlib.pyplot as plt
import seaborn as sns

class DataJobsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Definir palavras-chave para cada cargo
        self.job_titles = {
            'analista': ['analista de dados', 'data analyst', 'business analyst', 'analista bi'],
            'cientista': ['cientista de dados', 'data scientist', 'machine learning', 'ml engineer'],
            'engenheiro': ['engenheiro de dados', 'data engineer', 'arquiteto de dados', 'data architect']
        }
        
        # Skills técnicas comuns na área de dados
        self.skills_keywords = [
            'python', 'r', 'sql', 'pandas', 'numpy', 'matplotlib', 'seaborn',
            'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'spark', 'hadoop',
            'tableau', 'power bi', 'looker', 'qlik', 'excel', 'aws', 'azure',
            'gcp', 'docker', 'kubernetes', 'git', 'github', 'jupyter',
            'machine learning', 'deep learning', 'big data', 'etl',
            'data warehouse', 'mongodb', 'postgresql', 'mysql', 'oracle',
            'elasticsearch', 'kafka', 'airflow', 'dbt', 'snowflake',
            'redshift', 'bigquery', 'databricks', 'scala', 'java',
            'statistics', 'statistics', 'data visualization', 'business intelligence'
        ]
    
    def scrape_linkedin_jobs(self, job_query, max_pages=5):
        """Scraping de vagas do LinkedIn"""
        jobs_data = []
        
        for page in range(max_pages):
            try:
                url = f"https://www.linkedin.com/jobs/search?keywords={quote_plus(job_query)}&location=Brazil&start={page*25}"
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                job_cards = soup.find_all('div', class_='job-search-card')
                
                for card in job_cards:
                    try:
                        title = card.find('h3', class_='base-search-card__title')
                        if title:
                            job_title = title.text.strip()
                            
                        description_link = card.find('a')['href']
                        
                        # Buscar descrição detalhada
                        job_response = requests.get(description_link, headers=self.headers)
                        job_soup = BeautifulSoup(job_response.content, 'html.parser')
                        
                        description_div = job_soup.find('div', class_='show-more-less-html__markup')
                        description = description_div.text if description_div else ""
                        
                        jobs_data.append({
                            'title': job_title,
                            'description': description,
                            'source': 'LinkedIn'
                        })
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        continue
                        
                time.sleep(2)
                
            except Exception as e:
                print(f"Erro na página {page}: {e}")
                continue
                
        return jobs_data
    
    def scrape_indeed_jobs(self, job_query, max_pages=5):
        """Scraping de vagas do Indeed"""
        jobs_data = []
        
        for page in range(max_pages):
            try:
                start = page * 10
                url = f"https://br.indeed.com/jobs?q={quote_plus(job_query)}&l=Brasil&start={start}"
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                job_cards = soup.find_all('div', class_='job_seen_beacon')
                
                for card in job_cards:
                    try:
                        title_element = card.find('h2', class_='jobTitle')
                        if title_element:
                            job_title = title_element.text.strip()
                        
                        summary_element = card.find('div', class_='summary')
                        description = summary_element.text if summary_element else ""
                        
                        jobs_data.append({
                            'title': job_title,
                            'description': description,
                            'source': 'Indeed'
                        })
                        
                    except Exception as e:
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                print(f"Erro na página {page}: {e}")
                continue
        
        return jobs_data
    
    def extract_skills_from_text(self, text):
        """Extrair skills do texto da vaga"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def categorize_job(self, job_title, job_description):
        """Categorizar a vaga por tipo de profissional"""
        text = (job_title + " " + job_description).lower()
        
        # Pontuação para cada categoria
        scores = {'analista': 0, 'cientista': 0, 'engenheiro': 0}
        
        for category, keywords in self.job_titles.items():
            for keyword in keywords:
                if keyword in text:
                    scores[category] += 1
        
        # Retornar categoria com maior pontuação
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'outros'
    
    def run_full_scraping(self):
        """Executar scraping completo"""
        all_jobs = []
        
        # Queries de busca mais específicas
        search_queries = [
            "analista de dados",
            "cientista de dados", 
            "engenheiro de dados",
            "data analyst",
            "data scientist",
            "data engineer"
        ]
        
        print("Iniciando scraping das vagas...")
        
        for query in search_queries:
            print(f"Buscando: {query}")
            
            # Indeed
            try:
                indeed_jobs = self.scrape_indeed_jobs(query, max_pages=3)
                all_jobs.extend(indeed_jobs)
                print(f"  - Indeed: {len(indeed_jobs)} vagas")
            except Exception as e:
                print(f"  - Erro no Indeed: {e}")
            
            time.sleep(3)
        
        print(f"Total de vagas coletadas: {len(all_jobs)}")
        return all_jobs
    
    def analyze_skills(self, jobs_data):
        """Analisar skills por categoria profissional"""
        skills_by_category = {'analista': [], 'cientista': [], 'engenheiro': []}
        
        for job in jobs_data:
            category = self.categorize_job(job['title'], job['description'])
            if category in skills_by_category:
                skills = self.extract_skills_from_text(job['title'] + " " + job['description'])
                skills_by_category[category].extend(skills)
        
        # Contar frequência das skills
        results = {}
        for category, skills_list in skills_by_category.items():
            skill_counts = Counter(skills_list)
            results[category] = skill_counts.most_common(15)  # Top 15 skills
        
        return results
    
    def save_results(self, results):
        """Salvar resultados em arquivos"""
        # Salvar em JSON
        with open('skills_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Salvar em CSV
        df_data = []
        for category, skills in results.items():
            for skill, count in skills:
                df_data.append({
                    'categoria': category,
                    'skill': skill,
                    'frequencia': count
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv('skills_analysis.csv', index=False, encoding='utf-8')
        
        return df

    def plot_results(self, results):
        """Criar visualizações dos resultados"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        categories = ['analista', 'cientista', 'engenheiro']
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            if category in results and results[category]:
                skills = [skill for skill, count in results[category][:10]]
                counts = [count for skill, count in results[category][:10]]
                
                axes[i].barh(skills, counts, color=color)
                axes[i].set_title(f'Top Skills - {category.title()} de Dados')
                axes[i].set_xlabel('Frequência')
                
                # Inverter ordem para mostrar do maior para menor
                axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('skills_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    scraper = DataJobsScraper()
    
    # Executar scraping
    jobs_data = scraper.run_full_scraping()
    
    # Analisar skills
    skills_analysis = scraper.analyze_skills(jobs_data)
    
    # Salvar resultados
    df = scraper.save_results(skills_analysis)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DA ANÁLISE ===")
    for category, skills in skills_analysis.items():
        print(f"\n{category.upper()} DE DADOS:")
        for skill, count in skills:
            print(f"  {skill}: {count}")
    
    # Criar visualizações
    scraper.plot_results(skills_analysis)
