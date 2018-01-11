from django.shortcuts import render


def perfeval_bar_chart_view(request, input_dict, output_dict, widget):
    for i in input_dict['eval_results']:
        try:
            i['accuracy'] = i['accuracy'] / 100.0
        except:
            pass
    return render(request, 'visualizations/eval_bar_chart.html',
                  {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})


def perfeval_to_table_view(request, input_dict, output_dict, widget):
    return render(request, 'visualizations/eval_to_table.html',
                  {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})


def perfeval_pr_space_view(request, input_dict, output_dict, widget):
    return render(request, 'visualizations/pr_space.html',
                  {'widget': widget, 'input_dict': input_dict, 'output_dict': output_dict})
