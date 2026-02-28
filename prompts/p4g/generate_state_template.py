"""
Prompt templates for P4G (PersuasionForGood) aspect promoters in COOPER:
- Attention
- Appeal
- Proposition

Each builder function takes the needed text (e.g. dialogue history) and
returns a ready-to-send prompt string.
"""

from textwrap import dedent


def build_attention_prompt(dialogue_history: str) -> str:
    """
    Build the Attention aspect prompt.

    Parameters
    ----------
    dialogue_history : str
        The raw dialogue history between Persuader and Persuadee.

    Returns
    -------
    str
        Full prompt for asking the model to generate attention-related questions.
    """
    return dedent(f"""
    P4G Attention
    <DialogueHistory>
    {dialogue_history}

    In the above dialogue, the Persuader is trying to persuade the Persuadee to donate
    to a charity called "Save the Children".
    To better capture the Persuadee's attention and motivate them to talk about the topic,
    the Persuader should build rapport and ask about the Persuadee's knowledge, opinions,
    expectations, or personal experiences related to charity.
    Please list three additional questions the Persuader could ask to either gather such
    information or attract the Persuadee's attention (each less than 20 words; do not
    repeat or closely paraphrase any question already present in the dialogue history).
    """).strip()


def build_appeal_prompt(dialogue_history: str,
                        summarization_previous_efforts: str) -> str:
    """
    Build the Appeal aspect prompt.

    Parameters
    ----------
    dialogue_history : str
        The raw dialogue history between Persuader and Persuadee.
    summarization_previous_efforts : str
        A short summary of the Persuader's previous persuasive efforts.

    Returns
    -------
    str
        Full prompt for asking the model to generate new persuasive appeals.
    """
    return dedent(f"""
    Appeal
    [DialogueHistory]
    {dialogue_history}

    [SummarizationOfPreviousEfforts]
    {summarization_previous_efforts}

    The above [DialogueHistory] is a conversation between a Persuader and a Persuadee
    about a charity called "Save the Children". The Persuader wants to change the
    Persuadee's opinion and donation decision.

    [TypicalPersuasionStrategies]
    1) credibility appeal (using credentials or organizational impact to build trust);
    2) donation information (concrete details about the donation task);
    3) logical appeal;
    4) emotional appeal;
    5) foot-in-the-door (start with a small request, then follow with larger ones);
    6) self-modeling (the Persuader first states their own intention to donate);
    7) personal story (short narratives illustrating someone's donation experience).

    Considering the [DialogueHistory] and [SummarizationOfPreviousEfforts],
    list five new ways for the Persuader to further convince the Persuadee,
    each explicitly using one of the [TypicalPersuasionStrategies]
    (each less than 20 words; do not repeat content already mentioned in
    the dialogue history).
    """).strip()


def build_proposition_prompt(dialogue_history: str) -> str:
    """
    Build the Proposition aspect prompt.

    Parameters
    ----------
    dialogue_history : str
        The raw dialogue history between Persuader and Persuadee.

    Returns
    -------
    str
        Full prompt for asking the model to generate donation propositions.
    """
    return dedent(f"""
    Proposition
    <DialogueHistory>
    {dialogue_history}

    The above dialogue is between a Persuader and a Persuadee about a charity
    called "Save the Children". For effective persuasion, the Persuader needs
    to ask for donations in an appropriate way and politely inquire about the
    Persuadee's attitude.
    List three different ways to make such a donation proposition appropriately
    (each less than 20 words).
    """).strip()
