import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';

// 가장 단순한 형태의 금액 편집 위젯
class PriceEditWidget extends StatefulWidget {
  final WishlistDetailResponse wish;
  final Function(int) onSave;

  const PriceEditWidget({
    Key? key, 
    required this.wish,
    required this.onSave,
  }) : super(key: key);

  @override
  State<PriceEditWidget> createState() => _PriceEditWidgetState();
}

class _PriceEditWidgetState extends State<PriceEditWidget> {
  late TextEditingController _controller;
  late FocusNode _focusNode;
  bool _isEditing = false;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(
      text: NumberFormat('#,###').format(widget.wish.productPrice)
    );
    _focusNode = FocusNode();
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _startEditing() {
    setState(() {
      _isEditing = true;
    });
    
    // 다음 프레임에서 포커스 설정
    WidgetsBinding.instance.addPostFrameCallback((_) {
      FocusScope.of(context).requestFocus(_focusNode);
    });
  }

  void _savePrice() {
    if (_controller.text.isNotEmpty) {
      final price = int.parse(_controller.text.replaceAll(',', ''));
      widget.onSave(price);
      setState(() {
        _isEditing = false;
      });
    }
  }

  void _cancelEditing() {
    setState(() {
      _controller.text = NumberFormat('#,###').format(widget.wish.productPrice);
      _isEditing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return _isEditing
      ? Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '상품 금액',
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.backgroundBlack, 
                fontWeight: FontWeight.w200
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _controller,
              focusNode: _focusNode,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                suffixText: '원',
                border: const OutlineInputBorder(),
              ),
              inputFormatters: [
                FilteringTextInputFormatter.digitsOnly,
              ],
              onChanged: (value) {
                if (value.isNotEmpty) {
                  final number = int.tryParse(value.replaceAll(',', '')) ?? 0;
                  final formatted = NumberFormat('#,###').format(number);
                  if (formatted != value) {
                    _controller.value = TextEditingValue(
                      text: formatted,
                      selection: TextSelection.collapsed(offset: formatted.length),
                    );
                  }
                }
              },
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                TextButton(
                  onPressed: _cancelEditing,
                  child: Text('취소'),
                ),
                TextButton(
                  onPressed: _savePrice,
                  child: Text('저장'),
                ),
              ],
            ),
          ],
        )
      : InkWell(
          onTap: _startEditing,
          child: Row(
            children: [
              Text(
                '상품 금액',
                style: AppTextStyles.bodySmall.copyWith(
                  color: AppColors.backgroundBlack, 
                  fontWeight: FontWeight.w200
                ),
              ),
              const SizedBox(width: 40),
              Expanded(
                child: Text(
                  '${NumberFormat('#,###').format(widget.wish.productPrice)} 원',
                  style: AppTextStyles.bodyMedium,
                ),
              ),
            ],
          ),
        );
  }
}

// 사용 예:
// 
// _buildProductPriceSection(WishlistDetailResponse wish) {
//   return Padding(
//     padding: const EdgeInsets.symmetric(vertical: 25),
//     child: PriceEditWidget(
//       wish: wish,
//       onSave: (price) {
//         _saveField('productPrice', price);
//       },
//     ),
//   );
// }