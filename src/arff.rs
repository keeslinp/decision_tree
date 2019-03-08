use crate::record::Record;
use lazy_static::lazy_static;
use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::{Regex, RegexBuilder};

#[derive(Debug)]
pub enum Class {
    Nominal(Vec<String>),
    // Max
    Continuous(usize),
}

#[derive(Debug)]
pub struct Arff {
    pub records: Vec<Record>,
    pub labels: Vec<String>,
    pub classes: Vec<Class>,
}

#[derive(Debug)]
pub enum ArffError {}

impl Arff {
    pub fn parse(contents: &str) -> Result<Self, ArffError> {
        lazy_static! {
            static ref ATTRIBUTE: Regex = RegexBuilder::new(r"^@attribute\s+(\S+).*")
                .case_insensitive(true)
                .build()
                .unwrap();
            static ref DATA: Regex = RegexBuilder::new("^@data")
                .case_insensitive(true)
                .build()
                .unwrap();
            static ref NOMINAL: Regex = RegexBuilder::new(r"\{(.*)\}")
                .case_insensitive(true)
                .build()
                .unwrap();
            static ref CLASS: Regex = RegexBuilder::new(r"([^,]+),?")
                .case_insensitive(true)
                .build()
                .unwrap();
        }
        let mut records: Vec<Record> = Vec::new();
        let mut labels = Vec::new();
        let mut classes: Vec<Class> = Vec::new();
        let mut data_section = false;
        for line in contents.lines() {
            if line.chars().next() != Some('%') {
                if data_section {
                    let values: Vec<usize> = line
                        .split(',')
                        .enumerate()
                        .map(|(index, value)| match &mut classes[index] {
                            Class::Nominal(classes) => {
                                classes.iter().position(|ref c| *c == value).unwrap()
                            }
                            Class::Continuous(ref mut max) => {
                                let continuous_value = value.parse::<f32>().unwrap().floor() as usize;
                                if continuous_value > *max {
                                    *max = continuous_value;
                                }
                                continuous_value
                            },
                        })
                        .collect();
                    let (class, features) = values.split_last().expect("empty line");
                    records.push(Record {
                        class: *class,
                        features: Vec::from(features),
                    });
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
                            let mut class_list: Vec<String> = CLASS
                                .captures_iter(&classes_raw)
                                .map(|captures| {
                                    String::from(captures.get(1).unwrap().as_str().trim())
                                })
                                .collect();
                            class_list.push("?".to_owned());
                            classes.push(Class::Nominal(class_list));
                        } else {
                            classes.push(Class::Continuous(0));
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
