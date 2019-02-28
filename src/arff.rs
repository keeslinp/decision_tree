use lazy_static::lazy_static;
use regex::{RegexBuilder, Regex};
use rand::thread_rng;
use rand::seq::SliceRandom;
use crate::record::Record;

#[derive(Debug)]
pub struct Arff {
    pub records: Vec<Record>,
    pub labels: Vec<String>,
    pub classes: Vec<Vec<String>>,
}

#[derive(Debug)]
pub enum ArffError {}

impl Arff {
    pub fn parse(contents: &str) -> Result<Self, ArffError> {
        lazy_static! {
            static ref ATTRIBUTE: Regex = RegexBuilder::new(r"^@attribute\s+(\S+).*").case_insensitive(true).build().unwrap();
            static ref DATA: Regex = RegexBuilder::new("^@data").case_insensitive(true).build().unwrap();
            static ref NOMINAL: Regex = RegexBuilder::new(r"\{(.*)\}").case_insensitive(true).build().unwrap();
            static ref CLASS: Regex = RegexBuilder::new(r"([^,]+),?").case_insensitive(true).build().unwrap();
        }
        let mut records: Vec<Record> = Vec::new();
        let mut labels = Vec::new();
        let mut classes: Vec<Vec<String>> = Vec::new();
        let mut data_section = false;
        for line in contents.lines() {
            if line.chars().next() != Some('%') {
                if data_section {
                    let values: Vec<usize> = line.split(',').enumerate().map(|(index, value)| classes[index].iter().position(|ref c| *c == value).unwrap_or(0)).collect();
                    let (class, features) = values.split_last().expect("empty line");
                    records.push(Record {class: *class, features: Vec::from(features)});
                } else {
                    // Check if it is an attribute line
                    if let Some(label) = ATTRIBUTE
                        .captures(line)
                        .and_then(|cap| cap.get(1))
                        .map(|m| String::from(m.as_str().trim()))
                    {
                        labels.push(label);
                        // Check if it is nominal
                        if let Some(classes_raw) = NOMINAL
                            .captures(line)
                            .and_then(|cap| cap.get(1))
                            .map(|m| String::from(m.as_str()))
                        {
                            let class_list: Vec<String> = CLASS
                                .captures_iter(&classes_raw)
                                .map(|captures| String::from(captures.get(1).unwrap().as_str().trim()))
                                .collect();
                            classes.push(class_list);
                        } else {
                            panic!("Non-nominal data type");
                        }
                    } else if DATA.is_match(line) {
                        data_section = true;
                    }
                }
            }
        }
        Ok(Self {
            records,
            labels,
            classes,
        })
    }

    pub fn shuffle(&mut self) {
        self.records.as_mut_slice().shuffle(&mut thread_rng());
    }
}

