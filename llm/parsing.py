import json
import re
from typing import Any, Optional, Union
import simple_icd_10 as icd

CODE_RE = re.compile(r"\b[A-Z][0-9]{1,2}(?:\.[0-9]+)?\b")  # tolerant: S6, C77, F33.0, etc.

def remove_code_fences(s: str) -> str:
    s = s.strip()
    # remove leading ```json / ``` and trailing ```
    s = re.sub(r"^\s*```(?:json)?\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def extract_json_object(s: str) -> str:
    """Take substring from first '{' to last '}' if present, otherwise return as-is."""
    start = s.find("{")
    if start == -1:
        return s
    end = s.rfind("}")
    if end == -1:
        return s[start:]  # possibly truncated; will try to balance later
    return s[start:end + 1]

def balance_brackets(s: str) -> str:
    """Append missing closing brackets/braces if the string is truncated."""
    pairs = {"{": "}", "[": "]"}
    stack = []
    in_str = False
    esc = False

    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack and pairs[stack[-1]] == ch:
                    stack.pop()

    for opener in reversed(stack):
        s += pairs[opener]
    return s

def repair_common_json_issues(s: str) -> str:
    # 1) Fix tokens like \"F33.0\" that appear OUTSIDE strings.
    #    Avoid lookbehind: keep the delimiter in a capture group.
    s = re.sub(r'([:,\[\{]\s*)\\\"', r'\1"', s)      # opening \" -> "
    s = re.sub(r'\\\"(\s*[,}\]])', r'"\1', s)        # closing \" -> "

    # 2) Insert missing commas between adjacent quoted strings:
    #    ["A01" "B02"] -> ["A01", "B02"]
    s = re.sub(r'"\s+"', '", "', s)

    # 3) Insert missing commas between JSON values/fields when separated by whitespace:
    #    ..."comment": "x" "answer": [...]  -> ..."comment": "x", "answer": [...]
    #    ...] "answer": ...                 -> ...], "answer": ...
    s = re.sub(r'(["\]\}])\s+(?=["\[\{])', r'\1, ', s)

    # 4) Quote bare ICD-like tokens inside arrays/dicts, e.g. [I31?, F33.0] -> ["I31", "F33.0"]
    def quote_bare_code(m: re.Match) -> str:
        tok = m.group(1).rstrip("?")
        # Optional normalization: S6 -> S06 (comment out if you don't want this)
        if re.fullmatch(r"[A-Z][0-9]", tok):
            tok = tok[0] + "0" + tok[1]
        return f'"{tok}"'

    s = re.sub(r'(?<!")\b([A-Z][0-9]{1,2}(?:\.[0-9]+)?\??)\b(?!")(?=\s*[,}\]])',
               quote_bare_code, s)

    # 5) Remove trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    # 6) Balance brackets/braces (keep your existing balance_brackets() call if you have it)
    # If you already have balance_brackets(s) in your code, leave it there.
    s = balance_brackets(s)
    return s

def regex_fallback_extract_answer_codes(s: str) -> list[str]:
    """
    If JSON is unrecoverable, extract codes from the 'answer' region.
    Works even for truncated outputs.
    """
    # try to narrow to answer section
    m = re.search(r'"answer"\s*:\s*', s)
    region = s[m.end():] if m else s
    codes = CODE_RE.findall(region)

    # normalize one-digit categories if present
    out = []
    for c in codes:
        if re.fullmatch(r"[A-Z][0-9]", c):
            c = c[0] + "0" + c[1]
        out.append(c)

    # preserve order, deduplicate
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def parse_key_best_effort(s: str, key: str = "answer") -> Optional[Union[list, dict]]:
    raw = remove_code_fences(s)
    raw = extract_json_object(raw)

    # try strict + a few repair passes
    for _ in range(3):
        try:
            data = json.loads(raw)
            val = data.get(key, None)
            if val is None:
                return None

            # normalize to list/dict outputs (no fragile .split(", "))
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                # if model returned answer as string, pull codes from it
                return regex_fallback_extract_answer_codes(val)
            return None
        except Exception:
            raw = repair_common_json_issues(raw)

    # last resort: regex extract
    return regex_fallback_extract_answer_codes(raw)

def safe_parse(json_str, key):
    try:
        return parse_key_best_effort(json_str, key)
    except Exception as e:
        print(e)
        print(json_str)
        return None

###########################################################################################

def add_quotation_marks_around_codes(input_str):
    # Use regex to find and quote unquoted list elements
    fixed_str = re.sub(
        r'(?<=\[)([^\[\]\"]+)(?=\])',  # Pattern to match unquoted elements between []
        lambda m: ','.join(f'"{x.strip()}"' for x in m.group(1).split(',')), 
        input_str
    )
    return fixed_str

def fix_json_quotes(input_str):
    # Replace unescaped quotes in string values while preserving existing JSON structure
    fixed_str = re.sub(
        r'(?<=": ")(.*?)(?=",)',  # Match between ": " and ",
        lambda m: m.group(0).replace('"', '\\"'),  # Escape inner quotes
        input_str
    )
    return fixed_str

def get_key(s, key):
    try:
        #parsing json
        # json_str = s.strip('```json\n').strip('```').strip()
        json_str = remove_code_fences(s)
        try:
            data = json.loads(json_str)
            
        except:
            try:
                json_str = add_quotation_marks_around_codes(json_str)
                data = json.loads(json_str)
                
            except:
                json_str = fix_json_quotes(json_str)
                data = json.loads(json_str)

        #getting key  
        if isinstance(data[key], list) or isinstance(data[key], dict):
            return data[key]
        elif isinstance(data[key], str):
            return data[key].strip("[]").split(", ")
        else:
            raise ValueError("{} of type {} is not valid, should be one of (list, str, dict)".format(data[key], type(data[key])))
    
    except Exception as e:
        print(e)
        print(json_str)
        return None
    
###########################################################################################
#parsing codes string 

def expand_icd10_range(code_range):
    """
    Expands an ICD-10 code range into individual codes, handling cross-letter ranges.
    Example: 
      "B95-B97" → ["B95", "B96", "B97"]
      "A10-B12" → ["A10"... "A99", "B00", ... "B12"]
    """
    # Validate basic format
    if not re.fullmatch(r"[A-Z]\d+-[A-Z]\d+", code_range):
        print(f"Invalid ICD-10 range format: {code_range}")
        return []

    start, end = code_range.split('-')
    # print("Code1:", start, "; Code2:", end)
    
    prefix1 = start[0]
    prefix2 = end[0]
    if prefix2 < prefix1:
        start, end = end, start
        prefix1, prefix2 = prefix2, prefix1
    
    num1_str = start[1:]
    num2_str = end[1:]
    num1 = int(num1_str)
    num2 = int(num2_str)
    
    # If same letter prefix - simple case
    if prefix1 == prefix2:
        if num1 > num2:
            # raise ValueError(f"Invalid number order in range: {code_range}")
            num1, num2 = num2, num1
            num1_str, num2_str = num2_str, num1_str
        result = [f"{prefix1}{num:0{len(num1_str)}d}" for num in range(num1, num2 + 1)]
        result = [x for x in result if icd.is_category(x)]
        return result
    
    # Handle cross-letter ranges (A10-B12 etc.)
    result = []
    
    # 1. First part (start letter to Z99)
    first_letter_last_num = 99 if len(num1_str) == 2 else 9  # Handle different digit lengths
    result += [f"{prefix1}{num:0{len(num1_str)}d}" 
               for num in range(num1, first_letter_last_num + 1)]
    
    # 2. Middle letters (if any)
    for letter_ord in range(ord(prefix1) + 1, ord(prefix2)):
        letter = chr(letter_ord)
        first_num = 0
        last_num = 99 if len(num1_str) == 2 else 9
        result += [f"{letter}{num:0{len(num1_str)}d}" 
                   for num in range(first_num, last_num + 1)]
    
    # 3. Last part (A00 to end number)
    last_letter_first_num = 0
    result += [f"{prefix2}{num:0{len(num2_str)}d}" 
               for num in range(last_letter_first_num, num2 + 1)]
    
    result = [x for x in result if icd.is_category(x)]
    return result

def is_valid_icd10_string(input_str):
    # Split by commas and remove whitespace
    elements = [elem.strip() for elem in input_str.split(",")]
    
    # Regex patterns
    single_code_pattern = r'^[A-Z][0-9][0-9A-Z]?$'  # e.g., I01, A46
    range_pattern = r'^[A-Z][0-9][0-9A-Z]?-[A-Z][0-9][0-9A-Z]?$'  # e.g., I05-I09
    
    for elem in elements:
        if not (re.fullmatch(single_code_pattern, elem) or re.fullmatch(range_pattern, elem)):
            return False
    return True

n_bad_codes = 0
def parse_codes(codes):
    result = []
    # invalid_row = False
    for code in codes:
        if is_valid_icd10_string(code) and len(code.split(", ")) > 1:
            result += parse_codes(code.split(", "))
            
        elif "-" in code:
            result += expand_icd10_range(code)
            
        elif icd.is_valid_item(code):
            if icd.is_category(code):
                result.append(code)
            elif icd.is_subcategory(code):
                result.append(icd.get_parent(code))
            # elif icd.is_block(code):
            #     result += expand_icd10_range(code)
            else:
                print(f"{code} is valid ICD-10 item, but unknown")
                # # num_invalid_items += 1
                # invalid_row = True
            
        else:
            print(f"{code} is not ICD-10 item")
            global n_bad_codes
            n_bad_codes += 1
            # num_invalid_items += 1
            # invalid_row = True
            # print(codes)
            
    # if invalid_row: num_invalid_rows += 1
    return list(set(result))