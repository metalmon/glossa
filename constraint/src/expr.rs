//! Simple mathematical expression evaluator for GOST formulas.
//!
//! Supported: +, -, *, /, comparisons (==, !=, <, <=, >, >=),
//! parentheses, unary minus, constants (pi, e), variables.
//!
//! Grammar (operator precedence):
//!   expr     → comparison ( ("==" | "!=" | "<" | "<=" | ">" | ">=") comparison )?
//!   comparison → term ( ("+" | "-") term )*
//!   term     → factor ( ("*" | "/") factor )*
//!   factor   → "-" factor | primary
//!   primary  → NUMBER | VARIABLE | "pi" | "e" | "(" expr ")"

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Var(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        if ch.is_ascii_digit() || ch == '.' {
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' {
                    num.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            let n: f64 = num
                .parse()
                .map_err(|_| format!("invalid number: '{num}'"))?;
            tokens.push(Token::Number(n));
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(match ident.as_str() {
                "pi" => Token::Number(std::f64::consts::PI),
                "e" => Token::Number(std::f64::consts::E),
                _ => Token::Var(ident),
            });
        } else {
            match ch {
                '+' => {
                    chars.next();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    chars.next();
                    tokens.push(Token::Minus);
                }
                '*' => {
                    chars.next();
                    tokens.push(Token::Star);
                }
                '/' => {
                    chars.next();
                    tokens.push(Token::Slash);
                }
                '(' => {
                    chars.next();
                    tokens.push(Token::LParen);
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RParen);
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Eq);
                    } else {
                        return Err("expected '=='".into());
                    }
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Neq);
                    } else {
                        return Err("expected '!='".into());
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Le);
                    } else {
                        tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Ge);
                    } else {
                        tokens.push(Token::Gt);
                    }
                }
                _ => {
                    chars.next();
                    return Err(format!("unexpected character: '{ch}'"));
                }
            }
        }
    }

    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn parse_expr(&mut self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        let left = self.parse_comparison(vars)?;

        if let Some(token) = self.peek() {
            match token {
                Token::Eq => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if (left - right).abs() < 1e-12 {
                        1.0
                    } else {
                        0.0
                    });
                }
                Token::Neq => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if (left - right).abs() >= 1e-12 {
                        1.0
                    } else {
                        0.0
                    });
                }
                Token::Lt => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if left < right { 1.0 } else { 0.0 });
                }
                Token::Le => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if left <= right { 1.0 } else { 0.0 });
                }
                Token::Gt => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if left > right { 1.0 } else { 0.0 });
                }
                Token::Ge => {
                    self.advance();
                    let right = self.parse_comparison(vars)?;
                    return Ok(if left >= right { 1.0 } else { 0.0 });
                }
                _ => {}
            }
        }

        Ok(left)
    }

    fn parse_comparison(&mut self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        let mut left = self.parse_term(vars)?;

        while let Some(token) = self.peek() {
            match token {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_term(vars)?;
                    left += right;
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_term(vars)?;
                    left -= right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_term(&mut self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        let mut left = self.parse_factor(vars)?;

        while let Some(token) = self.peek() {
            match token {
                Token::Star => {
                    self.advance();
                    let right = self.parse_factor(vars)?;
                    left *= right;
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_factor(vars)?;
                    if right == 0.0 {
                        return Err("division by zero".into());
                    }
                    left /= right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_factor(&mut self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        if self.peek() == Some(&Token::Minus) {
            self.advance();
            let val = self.parse_factor(vars)?;
            return Ok(-val);
        }

        match self.peek() {
            Some(Token::Number(n)) => {
                let val = *n;
                self.advance();
                Ok(val)
            }
            Some(Token::Var(name)) => {
                let val = vars
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("undefined variable: '{name}'"))?;
                self.advance();
                Ok(val)
            }
            Some(Token::LParen) => {
                self.advance();
                let val = self.parse_expr(vars)?;
                if self.peek() != Some(&Token::RParen) {
                    return Err("expected ')'".into());
                }
                self.advance();
                Ok(val)
            }
            Some(t) => Err(format!("unexpected token: {:?}", t)),
            None => Err("unexpected end of expression".into()),
        }
    }
}

/// Evaluate a mathematical expression and return a boolean result.
/// Comparisons return 1.0 (true) or 0.0 (false); this treats 1.0 as true, 0.0 as false.
pub fn eval_bool(expr: &str, vars: &HashMap<String, f64>) -> Result<bool, String> {
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        return Err("empty expression".into());
    }
    let mut parser = Parser::new(tokens);
    let result = parser.parse_expr(vars)?;
    if parser.peek().is_some() {
        return Err(format!(
            "unexpected tokens after expression: {:?}",
            parser.tokens[parser.pos..]
                .iter()
                .map(|t| format!("{t:?}"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    Ok((result - 1.0).abs() < 1e-12)
}

/// Evaluate a mathematical expression and return a numeric value.
pub fn eval(expr: &str, vars: &HashMap<String, f64>) -> Result<f64, String> {
    let tokens = tokenize(expr)?;
    if tokens.is_empty() {
        return Err("empty expression".into());
    }
    let mut parser = Parser::new(tokens);
    let result = parser.parse_expr(vars)?;
    if parser.peek().is_some() {
        return Err(format!(
            "unexpected tokens after expression: {:?}",
            parser.tokens[parser.pos..]
                .iter()
                .map(|t| format!("{t:?}"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_addition() {
        assert_eq!(eval("1 + 2", &HashMap::new()).unwrap(), 3.0);
    }

    #[test]
    fn operator_precedence() {
        assert_eq!(eval("2 + 3 * 4", &HashMap::new()).unwrap(), 14.0);
    }

    #[test]
    fn parentheses() {
        assert_eq!(eval("(2 + 3) * 4", &HashMap::new()).unwrap(), 20.0);
    }

    #[test]
    fn unary_minus() {
        assert_eq!(eval("-5", &HashMap::new()).unwrap(), -5.0);
    }

    #[test]
    fn variable_substitution() {
        let mut vars = HashMap::new();
        vars.insert("x".into(), 10.0);
        assert_eq!(eval("x + 5", &vars).unwrap(), 15.0);
    }

    #[test]
    fn comparison_eq() {
        assert!(eval_bool("3 == 3", &HashMap::new()).unwrap());
        assert!(!eval_bool("3 == 4", &HashMap::new()).unwrap());
    }

    #[test]
    fn comparison_lt() {
        assert!(eval_bool("3 < 5", &HashMap::new()).unwrap());
        assert!(!eval_bool("5 < 3", &HashMap::new()).unwrap());
    }

    #[test]
    fn comparison_le() {
        assert!(eval_bool("3 <= 3", &HashMap::new()).unwrap());
        assert!(eval_bool("3 <= 5", &HashMap::new()).unwrap());
        assert!(!eval_bool("5 <= 3", &HashMap::new()).unwrap());
    }

    #[test]
    fn comparison_gt() {
        assert!(eval_bool("5 > 3", &HashMap::new()).unwrap());
        assert!(!eval_bool("3 > 5", &HashMap::new()).unwrap());
    }

    #[test]
    fn division_by_zero() {
        assert!(eval("1 / 0", &HashMap::new()).is_err());
    }

    #[test]
    fn pi_constant() {
        assert!((eval("pi", &HashMap::new()).unwrap() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn complex_expression() {
        let mut vars = HashMap::new();
        vars.insert("a".into(), 1.5);
        vars.insert("b".into(), 0.5);
        // (a + b) * 2 <= 5
        assert!(eval_bool("(a + b) * 2 <= 5", &vars).unwrap());
        assert!(!eval_bool("(a + b) * 2 > 5", &vars).unwrap());
    }

    #[test]
    fn gost_formula_typical() {
        let mut vars = HashMap::new();
        vars.insert("A".into(), 10.0);
        vars.insert("B".into(), 5.0);
        vars.insert("C".into(), 2.0);
        // (A + B) / C <= 10
        assert!(eval_bool("(A + B) / C <= 10", &vars).unwrap());
        // (A - B) * 2 == 10
        assert!(eval_bool("(A - B) * 2 == 10", &vars).unwrap());
    }

    #[test]
    fn undefined_variable_error() {
        let vars = HashMap::new();
        assert!(eval("x + 1", &vars).is_err());
    }

    #[test]
    fn empty_expression_error() {
        assert!(eval("", &HashMap::new()).is_err());
    }

    #[test]
    fn neq_comparison() {
        assert!(eval_bool("3 != 4", &HashMap::new()).unwrap());
        assert!(!eval_bool("3 != 3", &HashMap::new()).unwrap());
    }

    #[test]
    fn tokenize_number() {
        let tokens = tokenize("123.45").unwrap();
        assert_eq!(tokens, vec![Token::Number(123.45)]);
    }

    #[test]
    fn tokenize_identifiers() {
        let tokens = tokenize("abc + def").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Var("abc".into()),
                Token::Plus,
                Token::Var("def".into())
            ]
        );
    }

    #[test]
    fn tokenize_parentheses() {
        let tokens = tokenize("(x)").unwrap();
        assert_eq!(
            tokens,
            vec![Token::LParen, Token::Var("x".into()), Token::RParen]
        );
    }
}
