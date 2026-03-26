"""
CRI-RSK Chatbot — Query transformation service.
Rewrites user queries to improve retrieval quality.
Uses OpenAI for query rewriting.
"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """Tu es un assistant spécialisé dans l'investissement au Maroc et le CRI (Centre Régional d'Investissement) de Rabat-Salé-Kénitra.

Reformule la question de l'utilisateur pour améliorer la recherche dans une base de connaissances vectorielle. Tu dois respecter scrupuleusement ces règles :
1. Corriger les fautes d'orthographe et la grammaire.
2. Développer les abréviations connues (SARL, SA, SNC, AMDI, etc.).
3. CONSERVER IMPÉRATIVEMENT TOUS LES MOTS CLÉS : Les noms propres, les villes (ex: Rabat, Kenitra), les secteurs (ex: Industriel, Pêche maritime, GIAC, etc) et le métier inscrits par l'utilisateur ! Ne généralise JAMAIS la question en supprimant ces précisions.
4. Si la question est déjà claire et précise (ex: une phrase complète avec sujet/verbe/complément), RENVOIE-LA TELLE QUELLE sans modification.
5. Garder la même langue que la question originale.
6. Si la question est en darija/dialecte marocain, reformule en arabe classique ou en français.

Exemples :
- "créer boite" → "Quelles sont les étapes pour créer une entreprise au Maroc ?"
- "docs SARL" → "Quels sont les documents requis pour la création d'une Société à Responsabilité Limitée (SARL) ?"
- "Je cherche des informations sur l’investissement à Rabat dans le domaine industriel." → "Je cherche des informations sur l’investissement à Rabat dans le domaine industriel." (Déjà clair)
- "incitations pour GIAC Pêche maritime" → "Quelles sont les incitations et subventions disponibles pour les entreprises du secteur GIAC Pêche maritime ?"

Réponds UNIQUEMENT avec la question reformulée, sans aucune explication."""


async def transform_query(
    query: str,
    llm_model: str = "gpt-5.4",
    api_key: str = "",
) -> str:
    """
    Transform a user query to improve retrieval.
    Returns the rewritten query.
    Falls back to the original query on error.
    """
    # Skip transformation for very short or very clear queries
    if len(query.strip()) < 5:
        return query

    try:
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=llm_model,
            instructions=QUERY_REWRITE_PROMPT,
            input=f"Question originale: {query}",
            temperature=0,
            max_output_tokens=1000,
        )

        rewritten = response.output_text.strip() if response.output_text else ""

        # Validation: If rewritten query is significantly shorter than original,
        # it might be truncated or failed.
        if len(rewritten) < len(query) * 0.5:
            logger.warning(f"Transformed query seems truncated ('{rewritten}'), using original")
            return query

        if rewritten:
            logger.info(
                f"Query transformed: '{query[:50]}' → '{rewritten[:50]}'"
            )
            return rewritten
        return query

    except Exception as e:
        logger.warning(f"Query transformation failed: {e}")
        return query
