import 'package:marshmellow/data/datasources/remote/quit_api.dart';
import 'package:marshmellow/data/models/cookie/quit/quit_model.dart';

class QuitRepository {
  final QuitApi _quitApi;

  QuitRepository(this._quitApi);

  Future<AverageSpendingData> getAverageSpending() async {
    final response = await _quitApi.getAverageSpending();
    return response.data;
  }

   Future<DelusionData> getAvailableAmount() async {
    final response = await _quitApi.getAvailableAmount();
    return response.data;
  }
}
