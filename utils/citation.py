def format_sources_from_docs(docs):
    """
    Given a list of dicts or LangChain Document objects with 'source' and optional 'page',
    return a list of formatted strings like "filename (page X)".
    """
    formatted = []
    for d in docs:
        if isinstance(d, dict):
            src = d.get("source", "unknown")
            page = d.get("page", None)
        else:
            src = getattr(d, "metadata", {}).get("source", "unknown")
            page = getattr(d, "metadata", {}).get("page", None)
        if page is not None:
            formatted.append(f"{src} (page {page})")
        else:
            formatted.append(f"{src}")
    return formatted
