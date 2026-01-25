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
use chrono::{Datelike, Months, NaiveDate, Days, Weekday, TimeDelta};
use itertools::{Itertools, iproduct};
use std::error::Error;

use plotters::prelude::*;
use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::style::ShapeStyle;

use serde::{Serialize, Deserialize};
use serde_json;

const COPPER_OIL_RATIO_TOP:f64 = 205.0;

#[derive(Serialize, Deserialize)]
struct PlotConfigData {
    plot_width: u32,
    plot_height: u32,
    
    title_font: String,
    title_size: f64,

    label_font: String,
    label_size: f64,

    x_label_area_size: i32,
    y_label_area_size: i32,

    legend_font: String,
    legend_size: f64,

    line1_color_str: String,
    line2_color_str: String,

    #[serde(skip)]
    line1_color: OnceCell<RGBColor>,
    #[serde(skip)]
    line2_color: OnceCell<RGBColor>,

    stroke_size: u32,
    short_period: i64,
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
                
                title_font: String::from("黑体"),
                title_size: 72.0,

                label_font: String::from("仿宋"),
                label_size: 40.0,

                x_label_area_size: 60,
                y_label_area_size: 120,

                legend_font: String::from("仿宋"),
                legend_size: 56.0,

                line1_color_str: String::from("RED"),
                line2_color_str: String::from("BLUE"),

                line1_color: OnceCell::new(),
                line2_color: OnceCell::new(),

                stroke_size: 1,

                short_period: 1095,
            },
            data_conf: DataConfigData {
                date_shift: vec![90, 150]
            }
        }
    }
}

macro_rules! str2color {
    ($e:expr; $($c:ident),+; $d:expr) => {
        match $e {
            $(stringify!($c) => $c,)+
            _ => $d,
        }
    };
}

trait DataConfig {
    fn date_shift(&self) -> &Vec<u64>;
}

impl DataConfig for DataConfigData {
    fn date_shift(&self) -> &Vec<u64> {
        &self.date_shift
    }
}

type Period = NonZero<i64>;

fn period2dt(days: Period) -> TimeDelta {
    TimeDelta::days(days.get())
}

trait PlotConfig {
    fn x_label_area_size(&self) -> i32;
    fn y_label_area_size(&self) -> i32;
    fn line1_color(&self) -> RGBColor;
    fn line2_color(&self) -> RGBColor;
    fn title_style(&self) -> (&str, f64);
    fn label_style(&self) -> (&str, f64);
    fn legend_style(&self) -> (&str, f64);
    fn plot_size(&self) -> (u32, u32);
    fn plot_width(&self) -> u32;
    fn line1_stroke_style(&self) -> ShapeStyle;
    fn line2_stroke_style(&self) -> ShapeStyle;
    fn period(&self) -> Vec<Option<Period>>;
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

    fn line1_color(&self) -> RGBColor {
        self.line1_color.get_or_init(||PlotConfigData::str2color(&self.line1_color_str)).to_owned()
    }

    fn line2_color(&self) -> RGBColor {
        self.line2_color.get_or_init(||PlotConfigData::str2color(&self.line2_color_str)).to_owned()
    }

    fn title_style(&self) -> (&str, f64) {
        (self.title_font.as_str(), self.title_size)
    }

    fn label_style(&self) -> (&str, f64) {
        (self.label_font.as_str(), self.label_size)
    }

    fn legend_style(&self) -> (&str, f64) {
        (self.legend_font.as_str(), self.legend_size)
    }

    fn plot_size(&self) -> (u32, u32) {
        (self.plot_width, self.plot_height)
    }

    fn plot_width(&self) -> u32 {
        self.plot_width
    }

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

    fn period(&self) -> Vec<Option<Period>> {
        vec![None, NonZero::new(self.short_period)]
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

    fn as_f64_ref(&'_ self) -> Option<F64Column<'_>> {
        if let AnyData::F64(f) = &self.data {
            Some(F64Column { header: &self.header, data: f })
        } else {
            None
        }
    }

    fn as_date_ref(&'_ self) -> Option<DateColumn<'_>> {
        if let AnyData::DATE(d) = &self.data {
            Some(DateColumn { header: &self.header, data: d })
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

    fn to_date(c: &'_ DataColumn) -> Option<DateColumn<'_>> { c.as_date_ref()  }
    fn to_f64(c: &'_ DataColumn) -> Option<F64Column<'_>> { c.as_f64_ref() }
}


#[derive(Clone, Copy)]
struct F64Column<'a> {
    header: &'a str,
    data: &'a Vec<f64>,
}

#[derive(Clone, Copy)]
struct DateColumn<'a> {
    header: &'a str,
    data: &'a Vec<NaiveDate>,
}

impl<'a> IntoIterator for &'a F64Column<'a> {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> Deref for F64Column<'a> {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a> IntoIterator for &'a DateColumn<'a> {
    type Item = &'a NaiveDate;
    type IntoIter = std::slice::Iter<'a, NaiveDate>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> Deref for DateColumn<'a> {
    type Target = Vec<NaiveDate>;

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

    let copper_oil:Vec<DataColumn> = timedeltas.iter().copied()
        .map(dates_shift).chain(std::iter::once(copper_oil))
        .collect();

    Ok(MainData{copper_oil, indices})
}

#[derive(Clone)]
struct DateRange {
    start: NaiveDate,
    end: NaiveDate,
    short: bool,
}

impl DateRange {
    pub fn new(start_date:NaiveDate, end_date:NaiveDate, short_range:bool) -> DateRange {
        if start_date < end_date {
            DateRange { start: start_date, end: end_date, short: short_range }
        } else {
            DateRange { start: end_date, end: start_date, short: short_range }
        }
    }
}

impl Ranged for DateRange {
    type ValueType = NaiveDate;
    type FormatOption = DefaultFormatting;

    fn map(&self, v: &NaiveDate, pixel_range: (i32, i32)) -> i32 {
       let range_size = self.end - self.start;
       let dis = *v - self.start;
       let v = dis.num_days() as f64 / range_size.num_days() as f64;
       let size = pixel_range.1 - pixel_range.0;

       ((size as f64) * v).round() as i32 + pixel_range.0
    }

    fn key_points<Hint:KeyPointHint>(&self, hint: Hint) -> Vec<NaiveDate> {
        let start_year = self.start.year() + 1;
        let end_year = self.end.year() + 1;
        let years_cnt = (end_year - start_year).unsigned_abs() + 1;
        let months_cnt = (self.end -self.start).num_days().unsigned_abs() as u32 / 30 + 1;
        let max_num: u32 = hint.max_num_points().try_into().unwrap_or(u32::MAX);
        if max_num < 3 || years_cnt < 1 { return vec![]; } 

        let y2date = | y:i32 | NaiveDate::from_ymd_opt(y, 1, 1);

        if self.short && 6 * max_num > months_cnt {
            let too_few_points = |n: &u32| *n * max_num > months_cnt;
            let labelduration = [1u32, 2, 3, 4, 6].iter().copied().filter(too_few_points).min().expect("too many sample");

            let start_date = y2date(start_year - 1).expect("cannot?");
            let end_month = y2date(end_year).expect("end year error?!");

            let nth_point = |n: u32| start_date + Months::new(labelduration * n);
            return (0..months_cnt+12).map(nth_point).filter(floor(start_date)).filter(ceiling(end_month)).collect();
        } else {
            let step = years_cnt.div(max_num) as usize + 1_usize;
            return (start_year..end_year).step_by(step).filter_map(y2date).collect();
        }
    }

    fn range(&self) -> Range<NaiveDate> {
        self.start .. self.end
    }
}

type SampleData = (NaiveDate, f64);

fn first<T,U>(x: (T, U)) -> T { x.0 }
fn second<T,U>(x: (T, U)) -> U { x.1 }
fn accumulator<T: Add,U: Add>(x: (T, U), y: (T, U)) -> (T::Output, U::Output) {(x.0 + y.0, x.1 + y.1)}
fn to_ratio<T, U>(x: (T, U)) -> T::Output where T: Div<U> { x.0 / x.1 }

fn floor<T:PartialOrd>(min: T) -> impl Fn(&T) -> bool { move |x| *x >= min}
fn ceiling<T:PartialOrd>(max: T) -> impl Fn(&T) -> bool { move |x| *x <= max}
fn floor1st<T:PartialOrd, U>(min: T) -> impl Fn(&(T, U)) -> bool { move |x| x.0 >= min }
fn map1st<T,U,M>(f: impl Fn(&T) -> M) -> impl Fn(&(T, U)) -> M { move |x| f(&x.0) }

fn normalfilter(s: &f64) -> bool { s.is_normal() && *s > 0.0 }
fn samplefilter<T>(s: &(T, f64)) -> bool { normalfilter(&s.1) }

fn avg<I, T>(iter: I) -> Option<T>
where
    I: Iterator<Item = T>,
    T: Add<Output = T> + Div<f64, Output = T>,
{
    iter.map(|x| (x, 1.0)).reduce(accumulator).map(to_ratio)
}

fn series_lowfrq<I,F>(series: I, to:F) -> Vec<SampleData> 
where
    I: Iterator<Item = SampleData>,
    F: Fn(&NaiveDate) -> NaiveDate,
{
    series.filter(samplefilter)
        .chunk_by(map1st(to)).into_iter()
        .filter_map(|(d, gx)| avg(gx.map(second)).filter(normalfilter).map(|x|(d, x)) )
        .collect()
}

fn to_month(d: &NaiveDate) -> NaiveDate { NaiveDate::from_ymd_opt(d.year(), d.month(), 1).unwrap() }
fn to_week(d: &NaiveDate) -> NaiveDate { let w = d.iso_week(); NaiveDate::from_isoywd_opt(w.year(), w.week(), Weekday::Mon).unwrap() }

fn series4drawing<'a, I, R>(s1: &'a I, s2: &'a I, x_range: &R, w: u32) -> (Vec<SampleData>,Vec<SampleData>)
where
    R: Ranged<ValueType = NaiveDate>,
    &'a I: IntoIterator<Item = SampleData> + 'a,
{
    let x_range = x_range.range();
    let x_min = x_range.start;
    let x_duration = (x_range.end - x_range.start).num_days();
    let cor_series = s1.into_iter().filter(floor1st(x_min));
    let idx_series = s2.into_iter().filter(floor1st(x_min));

    let width = w as i64;
    if      x_duration < width   { (cor_series.collect(), idx_series.collect()) } 
    else if x_duration < width*7 { (series_lowfrq(cor_series, to_week),series_lowfrq(idx_series, to_week)) } 
    else                         { (series_lowfrq(cor_series, to_month),series_lowfrq(idx_series, to_month)) }
}

fn plot_range<'a, I>(s1: &'a I, s2: &'a I, duration: Option<Period>, w: u32)
    -> Result<(Vec<SampleData>, Vec<SampleData>, DateRange, Range<f64>, Range<f64>), Box<dyn Error>> 
where
    &'a I: IntoIterator<Item = SampleData> + 'a,
{
    let (min1, max1) = s1.into_iter().map(first).minmax().into_option().ok_or("copper oil date range")?;
    let (min2, max2) = s2.into_iter().map(first).minmax().into_option().ok_or("index date range")?;
    let x_max = max1.max(max2);
    let x_min = min1.max(min2);
    let x_min = duration.map(|days| x_min.max(x_max - period2dt(days))).unwrap_or(x_min);
    let x_range = DateRange::new(x_min, x_max, duration.is_some());

    let (yl_min, yl_max) = s2.into_iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("index range")?;
    let (yr_min, yr_max) = s1.into_iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("copper oil range")?;
    let yr_max = yr_max.min(COPPER_OIL_RATIO_TOP);
    let yl_range = yl_min * 0.99..yl_max * 1.01;
    let yr_range = yr_min * 0.99..yr_max * 1.01;

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
        = plot_range(s1, s2, duration, config.plot_width())?;
    println!("#size of series: {},{}", s1.len(), s2.len());

    let root = BitMapBackend::new(&file, config.plot_size()).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, config.title_style())
        .x_label_area_size(config.x_label_area_size())
        .y_label_area_size(config.y_label_area_size())
        .build_cartesian_2d(x_range.clone(), yl_range)?
        .set_secondary_coord(x_range, yr_range);

    chart.draw_series(LineSeries::new(s2,config.line2_stroke_style()))?
        .label(t2).legend(|(x, y)| Rectangle::new([(x,y-3),(x+32,y+3)], config.line2_color().filled()));

    chart.configure_mesh()
        .x_label_style(config.label_style())
        .x_label_formatter(&|x| if x.month() == 1 { format!("{:0}年", x.year()) } else { format!("{:0}月", x.month()) } )
        .y_label_style(config.label_style())
        .y_label_formatter(&|y| format!("{:0}", y))
        .draw()?;

    chart.draw_secondary_series(LineSeries::new(s1,config.line1_stroke_style()))?
        .label(t1).legend(|(x, y)| Rectangle::new([(x,y-3),(x+32,y+3)], config.line1_color().filled()));

    chart.configure_series_labels().position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK).background_style(WHITE).label_font(config.legend_style())
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

struct PointIter<'a> {
    x: DateColumn<'a>,
    y: F64Column<'a>,
    header: Cow<'a, str>,
}

impl<'a> PointIter<'a> {
    fn new<S: Into<Cow<'a, str>>>(header: S, x: DateColumn<'a>, y: F64Column<'a>)->PointIter<'a> 
    {
        assert!(x.len() == y.len(), "x序列与y序列长度不一致");
        PointIter{ x, y, header: header.into() } 
    }
}

impl<'a> Header for PointIter<'a> {
    fn get_header(&self) -> &str {
        &self.header
    }
}

impl<'a> IntoIterator for &'a PointIter<'a> {
    type Item = SampleData;
    type IntoIter = Box<dyn Iterator<Item = SampleData> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(zip_copied(&self.x, &self.y).filter(samplefilter))
    }
}

type PlotItem<'a> = (usize,(Option<Period>,&'a PointIter<'a>,&'a PointIter<'a>));

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
fn dateshift_header(x: &str, y: &str) -> String { format!("{}({})", y, x) }

fn multi_xy<'a, F, H>(data: &'a Vec<DataColumn>, header: F)
    -> Result<Vec<PointIter<'a>>, Box<dyn Error>>
where
    F: Fn(&'a str, &'a str) -> H,
    H: Into<Cow<'a, str>>,
{
    let result = iproduct!(data.iter().rev().filter_map(DataColumn::to_date),data.iter().rev().filter_map(DataColumn::to_f64))
        .map(|(x,y)|PointIter::new(header(x.header, y.header), x, y))
        .collect::<Vec<PointIter>>();

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

fn with_conf<'a, D, C>(conf: &'a C) -> impl Fn(D) -> (D, &'a C) {
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
        .map(with_conf(&conf.data_conf))
        .and_then(process_data)
        .map(with_conf(&conf.plot_conf))
        .and_then(plot_data)
}
