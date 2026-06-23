use regex::{Regex, RegexBuilder};

#[derive(Debug, Default, Clone)]
pub struct QueryOpts {
    pub ignore_case: bool,
    pub smart_case: bool,
    pub word: bool,
    pub fixed: bool,
}

pub fn compile(pattern: &str, opts: &QueryOpts) -> anyhow::Result<Regex> {
    let mut pat = if opts.fixed {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    if opts.word {
        pat = format!(r"\b(?:{})\b", pat);
    }
    let ci = opts.ignore_case
        || (opts.smart_case && !pattern.chars().any(|c| c.is_uppercase()));
    let re = RegexBuilder::new(&pat).case_insensitive(ci).build()?;
    Ok(re)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_strings_escape_metacharacters() {
        let re = compile("a.b", &QueryOpts { fixed: true, ..Default::default() }).unwrap();
        assert!(re.is_match("a.b"));
        assert!(!re.is_match("axb"));
    }

    #[test]
    fn word_boundaries_restrict_matches() {
        let re = compile("cat", &QueryOpts { word: true, ..Default::default() }).unwrap();
        assert!(re.is_match("the cat sat"));
        assert!(!re.is_match("category"));
    }

    #[test]
    fn smart_case_is_insensitive_for_lowercase_pattern() {
        let re = compile("cat", &QueryOpts { smart_case: true, ..Default::default() }).unwrap();
        assert!(re.is_match("Cat"));
    }

    #[test]
    fn smart_case_is_sensitive_when_pattern_has_uppercase() {
        let re = compile("Cat", &QueryOpts { smart_case: true, ..Default::default() }).unwrap();
        assert!(!re.is_match("cat"));
        assert!(re.is_match("Cat"));
    }

    #[test]
    fn ignore_case_forces_case_insensitive_even_with_uppercase_pattern() {
        let re = compile("Cat", &QueryOpts { ignore_case: true, ..Default::default() }).unwrap();
        assert!(re.is_match("cat"));
        assert!(re.is_match("CAT"));
    }
}
