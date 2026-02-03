/*
	copper_oil - Create Pictures of Copper Oil Ratio VS Stock Index

	Copyright(c) 2025 Luo Jie

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this softwareand associated documentation files(the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions :

	The above copyright noticeand this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
 */

use std::{borrow::Cow, cell::{Cell, LazyCell, OnceCell}, io::BufRead, num::NonZero, ops::{Add, Deref, Div, Range}};
use chrono::{Datelike, NaiveDate, Days};
use itertools::{Itertools, iproduct};
use std::error::Error;

use plotters::coord::ranged1d::{ValueFormatter, NoDefaultFormatting, KeyPointHint, Ranged};
use plotters::style::ShapeStyle;
use plotters::prelude::*;

use serde::{Serialize, Deserialize};
use serde_json;

#[derive(Serialize, Deserialize)]
struct PlotConfigData {
    plot_width: u32,
    plot_height: u32,
    bg_color_str: String,

    #[serde(skip)]
    bg_color: OnceCell<RGBColor>,

    title_font: String,
    title_size: f64,

    label_font: String,
    label_size: f64,

    x_label_area_size: i32,
    y_label_area_size: i32,

    legend_font: String,
    legend_size: f64,
    legend_rectangle: [i32; 4],
    legend_rect_bd_str: String,
    legend_rect_bg_str: String,

    #[serde(skip)]
    legend_rect_bd: OnceCell<RGBColor>,
    #[serde(skip)]
    legend_rect_bg: OnceCell<RGBColor>,


    line1_color_str: String,
    line2_color_str: String,

    #[serde(skip)]
    line1_color: OnceCell<RGBColor>,
    #[serde(skip)]
    line2_color: OnceCell<RGBColor>,

    stroke_size: u32,
    #[serde(default)]
    copper_oil_top: f64,
    #[serde(default)]
    short_period: i32,
}

#[derive(Serialize, Deserialize)]
struct DataConfigData {
    date_shift: Vec<u64>,
}

#[derive(Serialize, Deserialize)]
struct Config {
    data_conf: DataConfigData,
    plot_conf: PlotConfigData
}

impl Default for Config {
    fn default() -> Self {
        Config {
            plot_conf: PlotConfigData {
                plot_width: 1280,
                plot_height: 720,
                bg_color_str: String::from("WHITE"),
                bg_color: OnceCell::new(),
                
                title_font: String::from("黑体"),
                title_size: 72.0,

                label_font: String::from("仿宋"),
                label_size: 40.0,

                x_label_area_size: 60,
                y_label_area_size: 120,

                legend_font: String::from("仿宋"),
                legend_size: 56.0,
                legend_rectangle: [0, -3, 32, 3],
                legend_rect_bd_str: String::from("BLACK"),
                legend_rect_bg_str: String::from("WHITE"),
                legend_rect_bd: OnceCell::new(),
                legend_rect_bg: OnceCell::new(),

                line1_color_str: String::from("RED"),
                line2_color_str: String::from("BLUE"),
                line1_color: OnceCell::new(),
                line2_color: OnceCell::new(),

                stroke_size: 1,
                copper_oil_top: 205.0,
                short_period: 2,
            },
            data_conf: DataConfigData {
                date_shift: vec![90, 150]
            }
        }
    }
}

trait DataConfig {
    fn date_shift(&self) -> &Vec<u64>;
}

impl DataConfig for DataConfigData {
    fn date_shift(&self) -> &Vec<u64> {
        &self.date_shift
    }
}

type Period = NonZero<i32>;
type RectangleCorner = [i32; 4];

trait PlotConfig {
    fn x_label_area_size(&self) -> i32;
    fn y_label_area_size(&self) -> i32;
    fn line1_color(&self) -> RGBColor;
    fn line2_color(&self) -> RGBColor;
    fn title_style(&'_ self) -> TextStyle<'_>;
    fn label_style(&'_ self) -> TextStyle<'_>;
    fn legend_style(&'_ self) -> TextStyle<'_>;
    fn legend_rectangle(&self) -> RectangleCorner;
    fn legend_rect_bd(&self) -> RGBColor;
    fn legend_rect_bg(&self) -> RGBColor;
    fn plot_size(&self) -> (u32, u32);
    fn plot_width(&self) -> u32;
    fn bg_color(&self) -> RGBColor;
    fn line1_stroke_style(&self) -> ShapeStyle;
    fn line2_stroke_style(&self) -> ShapeStyle;
    fn copper_oil_top(&self) -> Option<f64>;
    fn period(&self) -> Vec<Option<Period>>;
}

macro_rules! str2color {
    ($e:expr; $($c:ident),+; $d:expr) => {
        match $e {
            $(stringify!($c) => $c,)+
            _ => $d,
        }
    };
}

macro_rules! fn_color {
    ($id:ident, $strid:ident) => {
        fn $id(&self) -> RGBColor {
            self.$id.get_or_init(||PlotConfigData::str2color(&self.$strid)).clone()
        }
    };
}

impl PlotConfigData {
    fn digi2color(color: &str) -> RGBColor {
        let color: Vec<u8> = color.split(",").filter_map(|s| s.trim().parse::<u8>().ok()).collect();
        if color.len() == 3 {RGBColor(color[0], color[1], color[2])} else {BLACK}
    }

    fn str2color(color: &str) -> RGBColor {
        str2color!(color; BLACK, BLUE, CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW; PlotConfigData::digi2color(color))
    }
}

impl PlotConfig for PlotConfigData {
    fn x_label_area_size(&self) -> i32 {
        self.x_label_area_size
    }

    fn y_label_area_size(&self) -> i32 {
        self.y_label_area_size
    }

    fn_color!(line1_color, line1_color_str);
    fn_color!(line2_color, line2_color_str);

    fn title_style(&'_ self) -> TextStyle<'_> {
        (self.title_font.as_str(), self.title_size).into()
    }

    fn label_style(&'_ self) -> TextStyle<'_> {
        (self.label_font.as_str(), self.label_size).into()
    }

    fn legend_style(&'_ self) -> TextStyle<'_> {
        (self.legend_font.as_str(), self.legend_size).into()
    }

    fn legend_rectangle(&self) -> RectangleCorner {
        self.legend_rectangle
    }

    fn_color!(legend_rect_bd, legend_rect_bd_str);
    fn_color!(legend_rect_bg, legend_rect_bg_str);

    fn plot_size(&self) -> (u32, u32) {
        (self.plot_width, self.plot_height)
    }

    fn plot_width(&self) -> u32 {
        self.plot_width
    }

    fn_color!(bg_color, bg_color_str);

    fn line1_stroke_style(&self) -> ShapeStyle {
        ShapeStyle {
            color: self.line1_color().mix(1.0),
            filled: true,
            stroke_width: self.stroke_size,
        }    
    }

    fn line2_stroke_style(&self) -> ShapeStyle {
        ShapeStyle {
            color: self.line2_color().mix(1.0),
            filled: true,
            stroke_width: self.stroke_size,
        }
    }

    fn copper_oil_top(&self) -> Option<f64> {
        (self.copper_oil_top != 0.0).then_some(self.copper_oil_top)
    }

    fn period(&self) -> Vec<Option<Period>> {
        let p = NonZero::new(self.short_period);
        if p.is_some() {vec![None, p]} else {vec![None]}
    }
}

enum AnyData {
    F64(Vec<f64>),
    DATE(Vec<NaiveDate>),
}

impl std::convert::From<Vec<f64>> for AnyData {
    fn from(value: Vec<f64>) -> Self {
        AnyData::F64(value)
    }
}

impl std::convert::From<Vec<NaiveDate>> for AnyData {
    fn from(value: Vec<NaiveDate>) -> Self {
        AnyData::DATE(value)
    }
}

enum AnyNum {
    F64(f64),
}

impl std::convert::From<f64> for AnyNum {
    fn from(value: f64) -> Self {
        AnyNum::F64(value)
    }
}

struct DataColumn {
    header: String,
    data: AnyData,
}

impl DataColumn {
    fn new(header: impl Into<String>, data: impl Into<AnyData>) -> DataColumn
    {
        DataColumn { header:header.into(), data: data.into() }
    }

    fn fromf64(header: impl Into<String>, data: Vec<f64>) -> DataColumn
    {
        DataColumn::new(header, data)
    }

    fn fromdate(header: impl Into<String>, data: Vec<NaiveDate>) -> DataColumn
    {
        DataColumn::new(header, data)
    }

    fn newf64(header: impl Into<String>) -> DataColumn
    {
        DataColumn { header:header.into(), data: AnyData::F64(vec![]) }
    }

    fn newdate(&mut self) {
        self.data = AnyData::DATE(vec![]);
    }

    fn as_f64_ref(&'_ self) -> Option<TypeColumn<'_, f64>> {
        if let AnyData::F64(f) = &self.data {
            Some(TypeColumn::<f64>{ header: &self.header, data: f })
        } else {
            None
        }
    }

    fn as_date_ref(&'_ self) -> Option<TypeColumn<'_, NaiveDate>> {
        if let AnyData::DATE(d) = &self.data {
            Some(TypeColumn::<NaiveDate>{ header: &self.header, data: d })
        } else {
            None
        }
    }

    fn push<S: AsRef<str>>(&mut self, item: &mut impl Iterator<Item = S>) -> Option<()> {
        let to_f64 = |f: S| f.as_ref().parse::<f64>().ok();
        let to_date  = |d: S| NaiveDate::parse_from_str(d.as_ref(), "%Y/%m/%d").ok();

        let item = item.next();
        match &mut self.data {
            AnyData::F64(f) => {
                f.push(item.and_then(to_f64).unwrap_or_default()); Some(())
            }
            AnyData::DATE(d) => {
                let push_to_dates = |i| d.push(i);
                item.and_then(to_date).map(push_to_dates)
            }
        }
    }

    fn to_date(c: &'_ DataColumn) -> Option<TypeColumn<'_, NaiveDate>> { c.as_date_ref()  }
    fn to_f64(c: &'_ DataColumn) -> Option<TypeColumn<'_, f64>> { c.as_f64_ref() }
}


#[derive(Clone, Copy)]
struct TypeColumn<'a, T> {
    header: &'a str,
    data: &'a Vec<T>,
}

impl<'a, T> IntoIterator for &'a TypeColumn<'a, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> Deref for TypeColumn<'a, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

struct MainData {
    copper_oil: Vec<DataColumn>,
    indices: Vec<DataColumn>,
}

fn read_data<P: AsRef<std::path::Path>>(input_file: &P)
    -> Result<MainData, Box<dyn Error>>
{
    let file = std::fs::File::open(&input_file)?;
    let lines = std::io::BufReader::new(file).lines();

    let mut indices: Vec<DataColumn> = Vec::with_capacity(6);

    lines.map(|l| l.expect("read file error!")).for_each(|line| {
        let mut item = line.split(",");
        if !indices.is_empty() {
            let push_item_into = |v: &mut DataColumn| v.push(&mut item);
            indices.iter_mut().map_while(push_item_into).count();
        } else {
            indices.extend(item.map(DataColumn::newf64));
            indices[0].newdate();
        }
    });

    if indices.len() > 3 {
        println!("#input item cnt: {}", indices.len());
        println!("#input item0 size: {}", indices[0].as_date_ref().unwrap().len());
        println!("#input itemE size: {}", indices.last().unwrap().as_f64_ref().unwrap().len());
        let copper_oil = indices.drain(1..=2).collect();
        Ok(MainData {copper_oil, indices})
    } else { 
        Err("too few items".into())
    }
}

fn process_data((data, config): (MainData, &impl DataConfig))
    -> Result<MainData, Box<dyn Error>>
{
    let copper_oil = data.copper_oil;
    let indices = data.indices;
    println!("#rawdata indices {}", indices.len() - 1);

    let dates = indices.first().and_then(DataColumn::to_date).ok_or("date incorrect")?;
    let copper  = copper_oil.last().and_then(DataColumn::to_f64).ok_or("copper incorrect")?;
    let oil = copper_oil.first().and_then(DataColumn::to_f64).ok_or("oil incorrect")?;
    let copper_oil = DataColumn::fromf64("铜油比", zip_copied(&copper,&oil).map(to_ratio).collect());
    
    let timedeltas = config.date_shift();

    let date_shift = |dt: u64| move |d: &NaiveDate| d.checked_add_days(Days::new(dt));
    let dates_shift = |dt| 
        DataColumn::fromdate(format!("推后{dt}天"), dates.iter().filter_map(date_shift(dt)).collect());

    let copper_oil:Vec<DataColumn> = timedeltas.iter().copied().map(dates_shift)
        .chain(std::iter::once(copper_oil))
        .collect();

    Ok(MainData{copper_oil, indices})
}

#[derive(Clone)]
struct DateRange<D: Datelike + PartialOrd + Copy> {
    start: D,
    end: D,
    short: bool,
}

impl<D: Datelike + PartialOrd + Copy> DateRange<D> {
    pub fn new(start_date:D, end_date:D, short_range:bool) -> DateRange<D> {
        if start_date < end_date {
            DateRange { start: start_date, end: end_date, short: short_range }
        } else {
            DateRange { start: end_date, end: start_date, short: short_range }
        }
    }
}

impl<D: Datelike + PartialOrd + Copy> Ranged for DateRange<D> {
    type ValueType = D;
    type FormatOption = NoDefaultFormatting;

    fn map(&self, v: &D, pixel_range: (i32, i32)) -> i32 {
       let range_size = self.end.num_days_from_ce() - self.start.num_days_from_ce();
       let dis = v.num_days_from_ce() - self.start.num_days_from_ce();
       let v = dis as f64 / range_size as f64;
       let size = pixel_range.1 - pixel_range.0;

       ((size as f64) * v).round() as i32 + pixel_range.0
    }

    fn key_points<Hint:KeyPointHint>(&self, hint: Hint) -> Vec<D> {
        let start_date = self.start.with_day(1).and_then(|d| d.with_month(1)).unwrap();
        let start_year = self.start.year() + 1;
        let end_year = self.end.year() + 1;
        let years_cnt = (end_year - start_year).unsigned_abs() + 1;
        let months_cnt = (self.end.num_days_from_ce() - self.start.num_days_from_ce()).unsigned_abs() as u32 / 30 + 1;
        let max_num: u32 = hint.max_num_points().try_into().unwrap_or(u32::MAX);
        if max_num < 3 || years_cnt < 1 { return vec![]; } 

        let y2date = | y:i32 | start_date.with_year(y);

        if self.short && 6 * max_num > months_cnt {
            let too_few_points = |n: &u32| *n * max_num > months_cnt;
            let labelduration = [1u32, 2, 3, 4, 6].iter().copied().filter(too_few_points).min().expect("too many sample");

            let end_month = start_date.with_year(end_year).expect("end year error?!");

            let nth_point = |n: u32| {
                let n = labelduration * n;
                let (dy, dm) = (n / 12, n % 12);
                start_date.with_year(start_date.year() + dy as i32).and_then(|y| y.with_month0(dm))
            };
            return (0..months_cnt+12).filter_map(nth_point).filter(floor(self.start)).filter(ceiling(end_month)).collect();
        } else {
            let step = years_cnt.div(max_num) as usize + 1_usize;
            return (start_year..end_year).step_by(step).filter_map(y2date).collect();
        }
    }

    fn range(&self) -> Range<D> {
        self.start .. self.end
    }
}

impl<D: Datelike + PartialOrd + Copy> ValueFormatter<D> for DateRange<D> {
    fn format(value: &D) -> String {
        if value.month() == 1 {format!("{:0}年", value.year())} else {format!("{:0}月", value.month())}
    }

    fn format_ext(&self, value: &D) -> String {
        Self::format(value)
    }
}

fn integer_fmt<T: std::fmt::Display>(v: &T) -> String { format!("{v:0}") }

type SampleData = (NaiveDate, f64);

fn first<T,U>(x: (T, U)) -> T { x.0 }
fn second<T,U>(x: (T, U)) -> U { x.1 }
fn accumulator<T: Add,U: Add>(x: (T, U), y: (T, U)) -> (T::Output, U::Output) {(x.0 + y.0, x.1 + y.1)}
fn to_ratio<T, U>(x: (T, U)) -> T::Output where T: Div<U> { x.0 / x.1 }

fn floor<T:PartialOrd>(min: T) -> impl Fn(&T) -> bool { move |x| *x >= min}
fn ceiling<T:PartialOrd>(max: T) -> impl Fn(&T) -> bool { move |x| *x <= max}
fn floor1st<T:PartialOrd, U>(min: T) -> impl Fn(&(T, U)) -> bool { move |x| x.0 >= min }
fn map1st<T,U,M>(f: impl Fn(&T) -> M) -> impl Fn(&(T, U)) -> M { move |x| f(&x.0) }

fn normalfilter<U: Into<AnyNum> + Copy>(s: &U) -> bool { match (*s).into() { AnyNum::F64(s) => s.is_normal() && s > 0.0 } }
fn samplefilter<T, U: Into<AnyNum> + Copy>(s: &(T, U)) -> bool { normalfilter(&s.1) }

fn avg<I, T>(iter: I) -> Option<T>
where
    I: Iterator<Item = T>,
    T: Add<Output = T> + Div<f64, Output = T>,
{
    iter.map(|x| (x, 1.0)).reduce(accumulator).map(to_ratio)
}

fn series_lowfrq<I,F,D,T>(series: I, to:F) -> Vec<(D,T)> 
where
    I: Iterator<Item = (D,T)>,
    F: Fn(&D) -> D,
    D: PartialEq,
    T: Into<AnyNum> + Add<Output = T> + Div<f64, Output = T> + Copy,
{
    series.filter(samplefilter)
        .chunk_by(map1st(to)).into_iter()
        .filter_map(|(d, gx)| avg(gx.map(second)).filter(normalfilter).map(|x|(d, x)) )
        .collect()
}

fn to_month<D: Datelike>(d: &D) -> D { d.with_day(1).unwrap() }
fn to_week<D: Datelike>(d: &D) -> D { let w = d.weekday().num_days_from_monday(); d.with_ordinal0(d.ordinal0().checked_sub(w).unwrap_or(0)).unwrap() }

fn series4drawing<'a, I, R, X, Y>(s1: &'a I, s2: &'a I, x_range: &R, w: u32) -> (Vec<(X,Y)>,Vec<(X,Y)>)
where
    R: Ranged<ValueType = X>,
    &'a I: IntoIterator<Item = (X,Y)> + 'a,
    X: Datelike + PartialOrd + Copy,
    Y: Into<AnyNum> + Add<Output = Y> + Div<f64, Output = Y> + Copy,
{
    let x_range = x_range.range();
    let x_min = x_range.start;
    let x_duration = x_range.end.num_days_from_ce() - x_range.start.num_days_from_ce();
    let cor_series = s1.into_iter().filter(floor1st(x_min));
    let idx_series = s2.into_iter().filter(floor1st(x_min));

    let width = w as i32;
    if      x_duration < width   { (cor_series.collect(), idx_series.collect()) } 
    else if x_duration < width*7 { (series_lowfrq(cor_series, to_week),series_lowfrq(idx_series, to_week)) } 
    else                         { (series_lowfrq(cor_series, to_month),series_lowfrq(idx_series, to_month)) }
}

fn plot_range<'a, I, X, Y>(s1: &'a I, s2: &'a I, duration: Option<Period>, yr_top: Option<Y>, w: u32)
    -> Result<(Vec<(X,Y)>, Vec<(X,Y)>, DateRange<X>, Range<Y>, Range<Y>), Box<dyn Error>> 
where
    &'a I: IntoIterator<Item = (X,Y)> + 'a,
    X: Datelike + PartialOrd + Copy,
    Y: Into<AnyNum> + Add<Output = Y> + Div<f64, Output = Y> + PartialOrd + Copy,
{
    let (min1, max1) = s1.into_iter().map(first).minmax().into_option().ok_or("copper oil date range")?;
    let (min2, max2) = s2.into_iter().map(first).minmax().into_option().ok_or("index date range")?;
    let x_max = if max1 > max2 {max1} else {max2};
    let x_min = if min1 > min2 {min1} else {min2};
    let x_min = duration.and_then(|dy| x_max.with_year(x_max.year() - dy.get()))
        .and_then(|x_start| (x_min < x_start).then_some(x_start)).unwrap_or(x_min);
    let x_range = DateRange::new(x_min, x_max, duration.is_some());

    let (yl_min, yl_max) = s2.into_iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("index range")?;
    let (yr_min, yr_max) = s1.into_iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("copper oil range")?;
    let yr_max = yr_top.and_then(|yr_top| (yr_max > yr_top).then_some(yr_top)).unwrap_or(yr_max);
    let yl_range = yl_min / 1.01..yl_max / 0.99;
    let yr_range = yr_min / 1.01..yr_max / 0.99;

    let (s1, s2) = series4drawing(s1, s2, &x_range, w);

    Ok((s1, s2, x_range, yl_range, yr_range))
}

fn plot<'a, P, C, I>(file: &P, config: &C, duration: Option<Period>, s1: &'a I, s2: &'a I)
    -> Result<(), Box<dyn Error>>
where
    C: PlotConfig,
    P: AsRef<std::path::Path>,
    I: Header,
    &'a I: IntoIterator<Item = SampleData> + 'a,
{
    let t1 = s1.get_header();
    let t2 = s2.get_header();
    let caption = format!("{t2}与{t1}比较",);

    let (s1, s2, x_range, yl_range, yr_range)
        = plot_range(s1, s2, duration, config.copper_oil_top(),config.plot_width())?;
    println!("#size of series: {},{}", s1.len(), s2.len());

    let legend_rect = move |x:i32, y:i32| {
        let corner = config.legend_rectangle();
        [(x + corner[0], y + corner[1]), (x + corner[2], y + corner[3])]
    };

    let root = BitMapBackend::new(&file, config.plot_size()).into_drawing_area();
    root.fill(&config.bg_color())?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, config.title_style())
        .x_label_area_size(config.x_label_area_size())
        .y_label_area_size(config.y_label_area_size())
        .build_cartesian_2d(x_range.clone(), yl_range)?
        .set_secondary_coord(x_range, yr_range);


    chart.draw_series(LineSeries::new(s2,config.line2_stroke_style()))?
        .label(t2).legend(|(x, y)| Rectangle::new(legend_rect(x,y), config.line2_color().filled()));

    chart.configure_mesh()
        .x_label_style(config.label_style())
        .y_label_style(config.label_style())
        .y_label_formatter(&integer_fmt)
        .draw()?;

    chart.draw_secondary_series(LineSeries::new(s1,config.line1_stroke_style()))?
        .label(t1).legend(|(x, y)| Rectangle::new(legend_rect(x,y), config.line1_color().filled()));

    chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft)
        .border_style(config.legend_rect_bd())
        .background_style(config.legend_rect_bg())
        .label_font(config.legend_style())
        .draw()?;

    root.present()?;

    Ok(())
}

fn zip_copied<'a, I1, I2, T, U>(x: &'a I1, y: &'a I2) -> impl Iterator<Item = (T, U)>
where 
    &'a I1: IntoIterator<Item = &'a T>,
    &'a I2: IntoIterator<Item = &'a U>,
    T: Copy + 'a, 
    U: Copy + 'a,
{
    x.into_iter().copied().zip(y.into_iter().copied())
}

trait Header {
    fn get_header(&self) -> &str;
}

struct PointIter<'a, X: Copy, Y: Into<AnyNum> + Copy> {
    x: TypeColumn<'a, X>,
    y: TypeColumn<'a, Y>,
    header: Cow<'a, str>,
}

impl<'a, X: Copy, Y: Into<AnyNum> + Copy> PointIter<'a, X, Y> {
    fn new<S: Into<Cow<'a, str>>>(header: S, x: TypeColumn<'a, X>, y: TypeColumn<'a, Y>)->PointIter<'a, X, Y> 
    {
        assert!(x.len() == y.len(), "x序列与y序列长度不一致");
        PointIter{ x, y, header: header.into() } 
    }
}

impl<'a, X: Copy, Y: Into<AnyNum> + Copy> Header for PointIter<'a, X, Y> {
    fn get_header(&self) -> &str {
        &self.header
    }
}

impl<'a, X: Copy, Y: Into<AnyNum> + Copy> IntoIterator for &'a PointIter<'a, X, Y> {
    type Item = (X, Y);
    type IntoIter = Box<dyn Iterator<Item = (X, Y)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(zip_copied(&self.x, &self.y).filter(samplefilter))
    }
}

type SeriesIter<'a> = PointIter<'a, NaiveDate, f64>;

type PlotItem<'a> = (usize,(Option<Period>,&'a SeriesIter<'a>,&'a SeriesIter<'a>));

fn except_index<S: AsRef<str>>(name: S) -> impl Fn(&PlotItem) -> bool
{
    move |(_, (_, _, s2)): &PlotItem| {
        s2.header != name.as_ref()
    }
}

fn skip_cnt(counter: &Cell<usize> ,f: impl Fn(&PlotItem) -> bool)
    -> impl Fn(&PlotItem) -> bool
{
    move |item: &PlotItem| {
        if f(item) {true} else {counter.set(counter.get() + 1);false}
    }
}

fn index_header<'a>(_: &str, y: &'a str) -> &'a str { y }
fn dateshift_header(x: &str, y: &str) -> String { format!("{y}({x})") }

fn multi_xy<'a, F, H>(data: &'a Vec<DataColumn>, header: F)
    -> Result<Vec<SeriesIter<'a>>, Box<dyn Error>>
where
    F: Fn(&'a str, &'a str) -> H,
    H: Into<Cow<'a, str>>,
{
    let result = iproduct!(data.iter().rev().filter_map(DataColumn::to_date),data.iter().rev().filter_map(DataColumn::to_f64))
        .map(|(x,y)|PointIter::new(header(x.header, y.header), x, y))
        .collect::<Vec<SeriesIter>>();

    if result.len() > 0 {Ok(result)} else {Err("main data error".into())}
}

fn plot_data((data, config): (MainData, &impl PlotConfig))
    -> Result<(), Box<dyn Error>>
{
    let copper_oil = multi_xy(&data.copper_oil, dateshift_header)?;
    let indices = multi_xy(&data.indices, index_header)?;
    let shortaxis = config.period();

    let num = indices.len() * copper_oil.len() * shortaxis.len();
    let skip_counter = Cell::new(0_usize);

    let plotter = |(n, (duration, s1, s2)): PlotItem| {
        let file = format!("copper_oil_{:02}.png",num - n);
        let print_err = |e: &Box<dyn Error>| eprintln!("plot {file} err {e}!");
        plot(&file, config, duration, s1, s2).inspect_err(print_err).ok()
    };

    let cnt = 
    iproduct!(shortaxis.iter().copied(), copper_oil.iter(), indices.iter())
        .enumerate()
        .filter(skip_cnt(&skip_counter, except_index("恒生指数")))
        .filter_map(plotter).count();

    println!("{cnt}/{num} files generated, {} files skipped, {} files error!",
                skip_counter.get(), num - cnt - skip_counter.get());
    Ok(())
}

fn load_conf_file<P: AsRef<std::path::Path>>(path: P) -> Option<Config> {
    let json2conf = |conf_str: String| serde_json::from_str(&conf_str).ok();
    std::fs::read_to_string(path).ok().and_then(json2conf)
}

fn with<'a, D, C>(conf: &'a C) -> impl Fn(D) -> (D, &'a C) {
    move |data: D| (data, conf)
}

pub fn read_and_plot_data(mut args: impl Iterator<Item = String>)
    -> Result<(), Box<dyn Error>>
{
    let load_conf_default = ||load_conf_file(".\\config.json");
    let data_path = args.next();
    let conf_path = args.next();
    let conf = LazyCell::new(||conf_path.as_ref()
        .and_then(load_conf_file)
        .or_else(load_conf_default)
        .unwrap_or_default());

    data_path.as_ref().ok_or("too few argument".into())
        .and_then(read_data)
        .map(with(&conf.data_conf))
        .and_then(process_data)
        .map(with(&conf.plot_conf))
        .and_then(plot_data)
}
