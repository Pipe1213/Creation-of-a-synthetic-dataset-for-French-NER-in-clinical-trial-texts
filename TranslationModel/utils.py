def get_criteria_str_list(text):
    """
    Get a list of sentences associated to each criteria from a given text
    in BIO format
    params:
        text: str
    return:
        criteria_str_list: list<str>
    """
    criteria_str_list = []
    # split by criteria (each individual criteria information is separated by an
    # empty line)
    criteria_list = text.split("\n\n")
    for criteria in criteria_list:
        criteria_lines = criteria.split("\n")
        criteria_str = ""
        strs = [line.split()[0] for line in criteria_lines if line != ""]
        criteria_str = " ".join(strs)
        criteria_str_list.append(criteria_str)
    return criteria_str_list
        
        