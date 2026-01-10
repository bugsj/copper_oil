use std::{fs::File, io::BufRead, path::Path, ops::{Add, Div, Range, Deref}, cell::Cell, borrow::Cow};
use chrono::{Datelike, Months, NaiveDate, TimeDelta, Weekday};
use itertools::{Itertools, iproduct};

use plotters::prelude::*;
use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::style::ShapeStyle;

const SHORT_DURATION:Months = Months::new(36);
const COPPER_OIL_RATIO_TOP:f64 = 205.0;

struct PlotConfig {
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

    line1_color: RGBColor,
    line2_color: RGBColor,

    stroke_size: u32,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
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

            line1_color: BLUE,
            line2_color: RED,

            stroke_size: 1,
        }
    }
}

impl PlotConfig {
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

    fn line1_stroke_style(&self) -> ShapeStyle {
        ShapeStyle {
            color: self.line1_color.mix(1.0),
            filled: true,
            stroke_width: self.stroke_size,
        }    
    }

    fn line2_stroke_style(&self) -> ShapeStyle {
        ShapeStyle {
            color: self.line2_color.mix(1.0),
            filled: true,
            stroke_width: self.stroke_size,
        }    
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
        let to_f64 = |s: Option<S>| s.and_then(|f| f.as_ref().parse::<f64>().ok());
        let to_date  = |s: Option<S>| s.and_then(|d| NaiveDate::parse_from_str(d.as_ref(), "%Y/%m/%d").ok());

        let item = item.next();
        match &mut self.data {
            AnyData::F64(f) => {
                f.push(to_f64(item).unwrap_or_default()); Some(())
            }
            AnyData::DATE(d) => { 
                to_date(item).map(|i| d.push(i))
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

fn read_data<P: AsRef<Path>>(input_file: &P)
    -> Result<(Vec<DataColumn>,Vec<DataColumn>), Box<dyn std::error::Error>>
{
    let file = File::open(&input_file)?;
    let lines = std::io::BufReader::new(file).lines();

    let mut data_columns: Vec<DataColumn> = Vec::with_capacity(6);

    lines.map(|l| l.expect("read file error!")).for_each(|line| {
        let mut item = line.split(",");
        if !data_columns.is_empty() {
            data_columns.iter_mut().map_while(|v | v.push(&mut item)).count();
        } else {
            data_columns.extend(item.map(DataColumn::newf64));
            data_columns[0].newdate();
        }
    });

    if data_columns.len() > 3 {
        println!("#input item cnt: {}", data_columns.len());
        println!("#input item0 size: {}", data_columns[0].as_date_ref().unwrap().len());
        println!("#input itemE size: {}", data_columns.last().unwrap().as_f64_ref().unwrap().len());
        let copper_oil = data_columns.drain(1..=2).collect();
        Ok((data_columns, copper_oil))
    } else { 
        Err("too few items".into())
    }
}

fn process_data((indices,copper_oil): (Vec<DataColumn>,Vec<DataColumn>))
    -> Result<(Vec<DataColumn>, Vec<DataColumn>), Box<dyn std::error::Error>>
{
    println!("#rawdata indices {}", indices.len() - 1);

    let dates = indices.first().and_then(DataColumn::to_date).ok_or("date incorrect")?;
    let copper  = copper_oil.last().and_then(DataColumn::to_f64).ok_or("copper incorrect")?;
    let oil = copper_oil.first().and_then(DataColumn::to_f64).ok_or("oil incorrect")?;
    
    let timedeltas = vec![90, 150];

    let date_shift = |dt: i64| move |d: &NaiveDate| *d + TimeDelta::days(dt);
    let copper_oil_ratio:Vec<DataColumn> = timedeltas.iter().copied()
        .map(|dt| DataColumn::new(format!("推后{dt}天"), dates.iter().map(date_shift(dt)).collect::<Vec<NaiveDate>>()))
        .chain(std::iter::once(DataColumn::new("铜油比", zip_copied(&copper,&oil).map(to_ratio).collect::<Vec<f64>>())))
        .collect();

    Ok((copper_oil_ratio, indices))
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

        let y2date = | y:i32 | NaiveDate::from_ymd_opt(y, 1, 1).expect("years err");

        if self.short && 6 * max_num > months_cnt {
            let labelduration = [1u32, 2, 3, 4, 6].iter().copied().filter(|n| *n * max_num > months_cnt).min().expect("too many sample");

            let remain = (self.start.month() - 1) % labelduration;
            let start_date = if remain == 0 {self.start} else {self.start + Months::new(labelduration - remain)};

            let start_month = to_month(&start_date);
            let start_month = if start_month < self.start { start_month + Months::new(labelduration) } else { start_month };
            let end_month = y2date(end_year);

            return (0..months_cnt).map(|n| start_month + Months::new(labelduration * n))
                .filter(|d| *d <= end_month).collect();
        } else {
            let step = years_cnt.div(max_num) as usize + 1_usize;
            return (start_year..end_year).step_by(step).map(y2date).collect();
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

fn series4drawing(s1: &DataIter, s2: &DataIter, x_range: &DateRange, w: u32) -> (Vec<SampleData>,Vec<SampleData>)
{
    let x_min = x_range.start;
    let x_duration = (x_range.end - x_range.start).num_days();
    let cor_series = s1.iter().filter(floor1st(x_min));
    let idx_series = s2.iter().filter(floor1st(x_min));

    let width = w as i64;
    if      x_duration < width   { (cor_series.collect(), idx_series.collect()) } 
    else if x_duration < width*7 { (series_lowfrq(cor_series, to_week),series_lowfrq(idx_series, to_week)) } 
    else                         { (series_lowfrq(cor_series, to_month),series_lowfrq(idx_series, to_month)) }
}

fn plot_range(s1: &DataIter, s2: &DataIter, short:bool)
    -> Result<(DateRange, Range<f64>, Range<f64>), Box<dyn std::error::Error>> 
{
    let (min1, max1) = s1.iter().map(first).minmax().into_option().ok_or("copper oil date range")?;
    let (min2, max2) = s2.iter().map(first).minmax().into_option().ok_or("index date range")?;
    let x_max = max1.max(max2);
    let x_min = min1.max(min2);
    let x_min = if short { x_min.max(x_max - SHORT_DURATION) } else { x_min }; // short = late
    let x_range = DateRange::new(x_min, x_max, short);

    let (yl_min, yl_max) = s2.iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("index range")?;
    let (yr_min, yr_max) = s1.iter().filter(floor1st(x_min)).map(second).minmax().into_option().ok_or("copper oil range")?;
    let yr_max = yr_max.min(COPPER_OIL_RATIO_TOP);
    let yl_range = yl_min * 0.99..yl_max * 1.01;
    let yr_range = yr_min * 0.99..yr_max * 1.01;

    Ok((x_range, yl_range, yr_range))
}

fn plot<P: AsRef<Path>>(file: &P, config: &PlotConfig, short:bool, s1: &DataIter, s2: &DataIter)
    -> Result<(), Box<dyn std::error::Error>>
{
    let t1 = s1.header.as_ref();
    let t2 = s2.header.as_ref();
    let caption = format!("{t2}与{t1}比较",);

    let (x_range, yl_range, yr_range) = plot_range(s1, s2, short)?;

    let (s1,s2) = series4drawing(s1, s2, &x_range, config.plot_width);
    println!("#size of series: {},{}", s1.len(), s2.len());

    let root = BitMapBackend::new(&file, config.plot_size()).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, config.title_style())
        .x_label_area_size(config.x_label_area_size)
        .y_label_area_size(config.y_label_area_size)
        .build_cartesian_2d(x_range.clone(), yl_range)?
        .set_secondary_coord(x_range, yr_range);

    chart.draw_series(LineSeries::new(s2,config.line2_stroke_style()))?
        .label(t2).legend(|(x, y)| Rectangle::new([(x,y-3),(x+32,y+3)], config.line2_color.filled()));

    chart.configure_mesh()
        .x_label_style(config.label_style())
        .x_label_formatter(&|x| if x.month() == 1 { format!("{:0}年", x.year()) } else { format!("{:0}月", x.month()) } )
        .y_label_style(config.label_style())
        .y_label_formatter(&|y| format!("{:0}", y))
        .draw()?;

    chart.draw_secondary_series(LineSeries::new(s1,config.line1_stroke_style()))?
        .label(t1).legend(|(x, y)| Rectangle::new([(x,y-3),(x+32,y+3)], config.line1_color.filled()));

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

struct DataIter<'a> {
    x: DateColumn<'a>,
    y: F64Column<'a>,
    header: Cow<'a, str>,
}

impl<'a> DataIter<'a> {
    fn new<S: Into<Cow<'a, str>>>(header: S, x: DateColumn<'a>, y: F64Column<'a>)->DataIter<'a> 
    {
        assert!(x.len() == y.len(), "x序列与y序列长度不一致");
        DataIter{ x, y, header: header.into() } 
    }

    fn with_y(y: F64Column<'a>)->impl Fn(DateColumn<'a>)->DataIter<'a> 
    { move |x: DateColumn<'a>| DataIter::new(format!("{}({})",y.header,x.header), x, y) }

    fn with_x(x: DateColumn<'a>)->impl Fn(F64Column<'a>)->DataIter<'a> {
        move |y: F64Column<'a>| DataIter::new(y.header, x, y)
    }

    fn iter(&self) -> impl Iterator<Item = SampleData>
    { zip_copied(&self.x, &self.y).filter(samplefilter) }
}

type PlotItem<'a> = (usize,(bool,&'a DataIter<'a>,&'a DataIter<'a>));

fn except_index<S: AsRef<str>>(name: S)
    -> impl Fn(&PlotItem) -> bool
{
    move |(_, (_, _, s2)): &PlotItem| {
        s2.header != name.as_ref()
    }
}

// fn except_150long((_, (short, co, index)):&PlotItem) -> bool
// {
//     if !short && co.header.ends_with("150天") {false} else {true}
// }

fn skip_cnt(counter: &Cell<usize> ,f: impl Fn(&PlotItem) -> bool)
    -> impl Fn(&PlotItem) -> bool
{
    move |item: &PlotItem| {
        if f(item) {true} else {counter.set(counter.get() + 1);false}
    }
}

fn multi_x<'a>(data: &'a Vec<DataColumn>)
    -> Result<Vec<DataIter<'a>>, Box<dyn std::error::Error>>
{
    let y = data.last().and_then(DataColumn::to_f64).ok_or("multi_x y incorrect")?;
    Ok(data.iter().rev().filter_map(DataColumn::to_date).map(DataIter::with_y(y)).collect())
}

fn multi_y<'a>(data: &'a Vec<DataColumn>)
    -> Result<Vec<DataIter<'a>>, Box<dyn std::error::Error>>
{
    let x = data.first().and_then(DataColumn::to_date).ok_or("multi_y x incorrect")?;
    Ok(data.iter().rev().filter_map(DataColumn::to_f64).map(DataIter::with_x(x)).collect())
}

fn plot_data((copper_oil_ratio, indices): (Vec<DataColumn>, Vec<DataColumn>))
    -> Result<(), Box<dyn std::error::Error>>
{
    let copper_oil_ratio = multi_x(&copper_oil_ratio)?;
    let indices = multi_y(&indices)?;
    let shortaxis = vec![false, true];

    let num = indices.len() * copper_oil_ratio.len() * shortaxis.len();
    let skip_counter = Cell::new(0_usize);

    let config = PlotConfig::default();
    let plotter = |(n, (short, s1, s2)): PlotItem| {
        let file = format!("copper_oil_{:02}.png",num - n);
        plot(&file, &config, short, s1, s2)
            .inspect_err(|e| eprintln!("plot {file} err {e}!")).ok()
    };

    let cnt = 
    iproduct!(shortaxis.iter().copied(), copper_oil_ratio.iter(), indices.iter())
        .enumerate()
        .filter(skip_cnt(&skip_counter, except_index("恒生指数")))
        .filter_map(plotter).count();

    println!("{cnt}/{num} files generated, {} files skipped, {} files error!", skip_counter.get(), num - cnt - skip_counter.get());
    Ok(())
}

pub fn read_and_plot_data(mut args: impl Iterator<Item = String>)
    -> Result<(), Box<dyn std::error::Error>>
{
    args.next().as_ref().ok_or("too few argument".into())
        .and_then(read_data)
        .and_then(process_data)
        .and_then(plot_data)
}
