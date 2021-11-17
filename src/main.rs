use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;

use deepl_api::*;
use indicatif::ProgressBar;

#[derive(Serialize, Deserialize, Debug)]
struct Answer {
    text: String,
    answer_start: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct QA {
    question: String,
    id: String,
    answers: Option<Vec<Answer>>,
    plausible_answers: Option<Vec<Answer>>,
    is_impossible: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct QAWithContext {
    qas: Vec<QA>,
    context: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Data {
    title: String,
    paragraphs: Vec<QAWithContext>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SQuAD {
    version: String,
    data: Vec<Data>,
}

fn ja_translate(txt: Vec<&str>) -> Vec<String> {
    let mut to_translate = Vec::<String>::new();
    for &t in &txt {
        to_translate.push(String::from(t));
    }
    let deepl = DeepL::new(std::env::var("DEEPL_API_KEY").unwrap());
    let texts = TranslatableTextList {
        source_language: Some("EN".to_string()),
        target_language: "JA".to_string(),
        texts: to_translate,
    };
    let translated = deepl.translate(None, texts).unwrap();

    let mut ans = Vec::<String>::new();
    for t in &translated {
        ans.push(t.text.clone());
    }
    ans
}

fn main() {
    let file = File::open("data/dev-v2.0.json").unwrap();
    let reader = BufReader::new(file);

    let squad: SQuAD = serde_json::from_reader(reader).unwrap();

    let dataset = squad.data;
    let mut dataset_ja = Vec::<Data>::new();

    let pb = ProgressBar::new(dataset.len() as u64);
    for data in &dataset {
        pb.inc(1);
        let mut paragraphs_ja = Vec::<QAWithContext>::new();
        let title_ja = ja_translate(vec![&data.title])[0].clone();

        for qa_with_context in &data.paragraphs {
            let mut qas_ja = Vec::<QA>::new();
            let context_ja = ja_translate(vec![&qa_with_context.context])[0].clone();
            for qa in &(*qa_with_context).qas {
                let answers = if qa.is_impossible {
                    qa.plausible_answers.as_ref().unwrap()
                } else {
                    qa.answers.as_ref().unwrap()
                };

                let question_ja = ja_translate(vec![&qa.question])[0].clone();
                let mut answers_ja = Vec::<Answer>::new();

                let mut text_to_ja = Vec::<&str>::new();
                for answer in answers {
                    text_to_ja.push(&answer.text);
                }

                let text_ja = ja_translate(text_to_ja);

                for (i, answer) in answers.iter().enumerate() {
                    answers_ja.push(Answer {
                        text: text_ja[i].clone(),
                        answer_start: answer.answer_start,
                    });
                }

                if qa.is_impossible {
                    qas_ja.push(QA {
                        question: question_ja,
                        id: qa.id.clone(),
                        answers: None,
                        plausible_answers: Some(answers_ja),
                        is_impossible: qa.is_impossible,
                    });
                } else {
                    qas_ja.push(QA {
                        question: question_ja,
                        id: qa.id.clone(),
                        answers: Some(answers_ja),
                        plausible_answers: None,
                        is_impossible: qa.is_impossible,
                    });
                }
            }
            paragraphs_ja.push(QAWithContext {
                qas: qas_ja,
                context: context_ja,
            });
        }
        dataset_ja.push(Data {
            title: title_ja,
            paragraphs: paragraphs_ja,
        });
        pb.finish_with_message("done");
    }

    let squad_ja = SQuAD {
        version: "v2.0".to_string(),
        data: dataset_ja,
    };

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("data/ja/dev-v2.0.json")
        .unwrap();

    write!(&file, "{}", serde_json::to_string(&squad_ja).unwrap()).unwrap();
}
