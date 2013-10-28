var csvArray;
$(document).ready(function(){
    $('#files').on('change', function(e){
        readFile(this.files[0], function(e) {
            //manipulate with result...
			var inputsFiles = e.target.result.split(",");
			
			var seriesData = [];
			for(int i = 1; i<inputsFiles.length; i++) {
				//document.writeln(inputsFiles[i]);
				seriesData.push([inputsFiles[i],inputsFiles[i+1]]);
				i++;
			}
			var chart = $('#container').highcharts();
			//chart.series[0].setName(inputsFiles[0]);
			chart.series[0].setData(seriesData);
            //$('#output_field').text(e.target.result);
        });
        
    });
});

function readFile(file, callback){
    var reader = new FileReader();
    reader.onload = callback
    reader.readAsText(file);
}


$(document).ready(function() {

    var options = {
            chart: {
                renderTo: 'chart',
                defaultSeriesType: 'line',
                marginRight: 130,
                marginBottom: 25
            },
            title: {
                text: 'reading',
                x: -20      //center
            },
            xAxis: {
                title: {
                    text: 'Stock'
                },
                categories: []
            },
            yAxis: {
                title: {
                    text: 'reading'
                }
            },
            series: []
    };

    $.get('../test.csv', function(data) {

        // Split the lines using newline 'n' as delimiter
        var lines = data.split('n');

        $.each(lines, function(lineNo, line) {

            // split the line elements using comma ',' as delimiter
            var items = line.split(',');

            $.each (items, function(itemNo, item) {
                if(itemNo == 0)
                    options.xAxis.categories.push(item);
                else if(itemNo > 0) {
                    var series = {
                            data: []
                    };
                    series.data.push(parseFloat(item));
                };
                options.series.push(series);
            });
            var chart = new HighCharts.Chart(options);
        });
    });

});
// AJAX Template	
// $(document).ready(function() {          
    // var c = [];
    // var d = [];

    // $.get('data.csv', function(data) {
        // var lines = data.split('n');
        // $.each(lines, function(lineNo, line) {
            // var items = line.split(',');
            // c.push(items[0]);
            // d.push(parseInt(items[1]));
        // });
    // });

    // var options = {
        // chart: {
            // renderTo: 'chart',
            // defaultSeriesType: 'line'
        // },
        // title: {
            // text: 'reading'
        // },
        // xAxis: {
            // title: {
                // text: 'Date Measurement'
            // },
            // categories: c
        // },
        // yAxis: {
            // title: {
                // text: 'reading'
            // }
        // },
        // series: [{
            // data: d
        // }]
    // };

    // var chart = new Highcharts.Chart(options);

// });