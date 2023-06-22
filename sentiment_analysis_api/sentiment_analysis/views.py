from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from setfit import SetFitModel

# Load the pre-trained sentiment analysis model
model = SetFitModel.from_pretrained("StatsGary/setfit-ft-sentinent-eval")

@csrf_exempt
def analyze(request):
    if request.method == 'POST':
        try:
            # Get the text input from the request payload
            text = request.POST.get('text', '')
            # Perform sentiment analysis on the text
            preds = model([text])

            # Map the sentiment labels to 'positive', 'negative', or 'neutral'
            sentiment = 'positive' if preds[0] > 0 else 'negative' if preds[0] < 0 else 'neutral'

            # sentiment = 'positive' if preds[0] < 0.5 else 'negative' if preds[0] > 0.5 else "neutral"

            # Return the sentiment analysis result as a JSON response
            return JsonResponse({'sentiment': sentiment})

        except Exception as e:
            # Handle any errors and provide an appropriate error response
            return JsonResponse({'error': str(e)}, status=500)

    # Return an error response for invalid requests
    return JsonResponse({'error': 'Invalid request'}, status=400)