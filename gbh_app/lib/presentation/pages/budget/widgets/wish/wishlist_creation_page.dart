import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/wish/wish_list_input_widget.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:path/path.dart' as path;
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart' as intl;
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wishlist_providers.dart';

/*
  위시리스트 생성 페이지
*/
class WishlistCreationPage extends ConsumerStatefulWidget {
  final String? sharedUrl; // 공유된 URL

  const WishlistCreationPage({super.key, this.sharedUrl});

  @override
  ConsumerState<WishlistCreationPage> createState() => _WishlistCreationPageState();
}

class _WishlistCreationPageState extends ConsumerState<WishlistCreationPage> {
  final _formKey = GlobalKey<FormState>();
  
  // 컨트롤러 정의
  final _nickNameController = TextEditingController();
  final _productNameController = TextEditingController();
  final _priceController = TextEditingController();
  final _urlController = TextEditingController();

  // 이미지
  File? _selectedImage;
  bool _showInputFields = false;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    
    // 위시리스트 생성 상태 초기화
    Future.microtask(() {
      ref.read(wishlistCreationProvider.notifier).resetState();

      // widget.sharedUrl이 있다면 처리
      if (widget.sharedUrl != null && widget.sharedUrl!.isNotEmpty) {
        _urlController.text = widget.sharedUrl!;
        _processSharedUrl(widget.sharedUrl!);
      }
      
      // 공유된 URL이 있으면 처리
      if (widget.sharedUrl != null && widget.sharedUrl!.isNotEmpty) {
        setState(() {
          _urlController.text = widget.sharedUrl!;
        });
        _processSharedUrl(widget.sharedUrl!);
      }
    });
  }

  @override
  void dispose() {
    _nickNameController.dispose();
    _productNameController.dispose();
    _priceController.dispose();
    _urlController.dispose();
    super.dispose();
  }

  // 공유된 URL 처리
  Future<void> _processSharedUrl(String url) async {
    if (url.isEmpty) return;
    
    setState(() {
      _isLoading = true;
    });
    
    try {
      // URL 크롤링 API 호출
      final data = await ref.read(wishlistCreationProvider.notifier).crawlProductUrl(url);
      
      if (data != null) {
        setState(() {
          // 상품명 설정
          if (data['productName'] != null && data['productName'].isNotEmpty) {
            _productNameController.text = data['productName'];
            _nickNameController.text = data['productName']; // 상품명을 닉네임에도 설정
          }
          
          // 이미지 URL이 있으면 다운로드
          if (data['productImage'] != null && data['productImage'].isNotEmpty) {
            _downloadImage(data['productImage']);
          }
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('URL 정보를 가져오는 중 오류가 발생했습니다: $e')),
        );
      }
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // 이미지 URL에서 이미지 다운로드
  Future<void> _downloadImage(String imageUrl) async {
    if (imageUrl.isEmpty) return;
    
    // URL이 //로 시작하면 https: 추가
    if (imageUrl.startsWith('//')) {
      imageUrl = 'https:$imageUrl';
    }
    
    try {
      final response = await http.get(Uri.parse(imageUrl));
      
      if (response.statusCode == 200) {
        // 임시 디렉토리에 이미지 저장
        final tempDir = await Directory.systemTemp.createTemp();
        final tempFile = File('${tempDir.path}/temp_image.jpg');
        await tempFile.writeAsBytes(response.bodyBytes);
        
        if (mounted) {
          setState(() {
            _selectedImage = tempFile;
          });
        }
      }
    } catch (e) {
      print('이미지 다운로드 중 오류 발생: $e');
    }
  }

  // 이미지 선택 함수
  Future<void> _pickImage() async {
    try {
      final ImagePicker imagePicker = ImagePicker();
      final XFile? image = await imagePicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 800,
        imageQuality: 80,
      );

      if (image == null) return;

      setState(() {
        _selectedImage = File(image.path);
      });
    } catch (e) {
      // Handle error
      print('Error picking wishlist image: $e');
    }
  }

  // 금액 입력 포맷 함수
  void _formatPrice(String value) {
    if (value.isNotEmpty) {
      final number = int.tryParse(value.replaceAll(',', '')) ?? 0;
      final formatted = intl.NumberFormat('#,###').format(number);
      
      if (formatted != value) {
        _priceController.value = TextEditingValue(
          text: formatted,
          selection: TextSelection.collapsed(offset: formatted.length),
        );
      }
    }
  }

  // 위시리스트 생성 함수
  Future<void> _createWishlist() async {
    if (_formKey.currentState?.validate() ?? false) {
      try {
        final price = int.parse(_priceController.text.replaceAll(',', ''));

        // URL 형식 수정
        String formattedUrl = _urlController.text;
        if (formattedUrl.isNotEmpty && !formattedUrl.startsWith('http')) {
          formattedUrl = 'http://$formattedUrl';
        }

        // 위시리스트 생성 요청
        await ref.read(wishlistCreationProvider.notifier).createWishlist(
          productNickname: _nickNameController.text,
          productName: _productNameController.text,
          productPrice: price,
          productUrl: formattedUrl,
          imageFile: _selectedImage,              
        );

        // 상태 확인
        final state = ref.read(wishlistCreationProvider);

        if (!state.isLoading && state.errorMessage == null) {
           if (mounted) {
            CompletionMessage.show(context, message: '위시생성완!');
            ref.read(wishlistProvider.notifier).fetchWishlists();

            Future.delayed(const Duration(seconds: 2), () {
              if (Navigator.of(context).canPop()) {
                Navigator.of(context).pop(); // 이전 화면으로 복귀
              } else {
                Navigator.of(context).pushReplacementNamed('/home'); // 혹시 이전 경로 없을 경우
              }
            });
          }
        } else if (state.errorMessage != null) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(state.errorMessage!)),
            );
          }
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('위시리스트 생성 중 오류가 발생했습니다: $e')),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // 위시리스트 생성 상태 관찰
    final creationState = ref.watch(wishlistCreationProvider);
    final isLoading = _isLoading || creationState.isLoading;
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '위시리스트',
      ),
      body: isLoading
        ? const Center(child: CircularProgressIndicator())
        : SingleChildScrollView(
            child: Form(
              key: _formKey,
              child:_buildInputFields(),
            ),
          ),
    );
  }
  
  Widget _buildInputFields() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 이미지 있으면 이미지, 없으면 캐릭터
          if (_selectedImage != null)
            Center(
              child: Container(
                margin: const EdgeInsets.only(bottom: 20),
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  image: DecorationImage(
                    image: FileImage(_selectedImage!),
                    fit: BoxFit.cover,
                  ),
                ),
              ),
            )
          else
            Center(
              child: Image.asset(
                'assets/images/characters/char_lying_down.png',
                width: 100,
                height: 100,
              ),
            ),
            
          const SizedBox(height: 20),
          
          // 이름 입력
          WishlistInput(
            controller: _productNameController,
            label: '상품명',
            hintText: '상품명을 입력하세요',
            validator: (value) {
              if (value == null || value.isEmpty) {
                return '상품명은 필수 입력 항목입니다.';
              }
              return null;
            },
          ),
          
          const SizedBox(height: 12),
          
          // 상품 금액 입력
          WishlistInput(
            controller: _priceController,
            label: '상품 금액',
            hintText: '금액을 입력하세요',
            keyboardType: TextInputType.number,
            inputFormatters: [
              FilteringTextInputFormatter.digitsOnly,
            ],
            onChanged: (value) => _formatPrice(value),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return '상품 금액은 필수 입력 항목입니다.';
              }
              return null;
            },
            suffix: const Text('원', style: TextStyle(color: Colors.grey)),
          ),
          
          const SizedBox(height: 12),
          
          // URL 입력
          WishlistInput(
            controller: _urlController,
            label: '상품 URL',
            hintText: '상품의 링크가 있으면 넣어주세요',
          ),
          
          const SizedBox(height: 80),
          
          // 이미지 업로드 텍스트
          Center(
            child: Text(
              '위시의 별칭이 있나요?\n이미지도 있다면 넣어주세요!',
              style: AppTextStyles.bodySmall,
              textAlign: TextAlign.center,
            ),
          ),
          
          const SizedBox(height: 16),
          
          // 이미지 업로드 필드
          Row(
            children: [
              Expanded(
                child: WishlistInput(
                  controller: _nickNameController,
                  label: '위시 닉네임',
                  hintText: '상세 설명 (선택)',
                ),
              ),
              const SizedBox(width: 12),
              GestureDetector(
                onTap: _pickImage,
                child: Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: AppColors.whiteLight,
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color:AppColors.backgroundBlack,
                    ),
                  ),
                  child: Icon(
                    _selectedImage != null 
                        ? Icons.check_circle
                        : Icons.camera_alt,
                    color: _selectedImage != null 
                        ? AppColors.bluePrimary
                        : Colors.grey,
                  ),
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 24),
          
          // 저장 버튼
          Button(
            text: '저장',
            textColor: AppColors.whiteLight,
            onPressed: _createWishlist,
          ),
        ],
      ),
    );
  }
}