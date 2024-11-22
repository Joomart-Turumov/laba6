import arxiv
import pandas as pd

def collect_data(query="data science", max_results=100):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = search.results()
    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return pd.DataFrame(papers)
