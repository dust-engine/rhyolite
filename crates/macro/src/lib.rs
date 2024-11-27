use syn::{
    parse::{Parse, ParseBuffer, ParseStream},
    parse_macro_input,
    spanned::Spanned,
    visit_mut::VisitMut,
    Stmt, Token,
};

struct GPUFutureBlock {
    pub capture: Option<Token![move]>,
    pub stmts: Vec<Stmt>,
}
impl Parse for GPUFutureBlock {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(GPUFutureBlock {
            capture: input.parse()?,
            stmts: syn::Block::parse_within(input)?,
        })
    }
}

#[proc_macro]
pub fn gpu_future(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as GPUFutureBlock);
    let GPUFutureBlock { mut stmts, capture } = input;

    let mut traverser: GPUFutureTraverser = GPUFutureTraverser::default();
    for stmt in stmts.iter_mut() {
        traverser.visit_stmt_mut(stmt);
    }

    let retained_values =
        (0..traverser.retained_value_count).map(|_| syn::Token![_](proc_macro2::Span::call_site()));
    quote::quote! {
        async #capture {
            let mut __retained_values: ::std::mem::MaybeUninit<(#(#retained_values, )*)> =  ::std::mem::MaybeUninit::zeroed();
            let __returned_values = {
                #(#stmts)*
            };
            rhyolite::future::GPUFutureBlockReturnValue {
                output: __returned_values,
                retained_values: unsafe{__retained_values.assume_init()},
            }
        }
    }.into()
}

#[derive(Default)]
struct GPUFutureTraverser {
    retained_value_count: usize,
    is_in_divergent_control_flow: bool,
    is_in_loop: bool,
}

impl GPUFutureTraverser {
    fn transform_retain_macro(&mut self, inner: &syn::Macro) -> proc_macro2::TokenStream {
        let count = syn::Index::from(self.retained_value_count);
        self.retained_value_count += 1;
        let tokens = &inner.tokens;

        if self.is_in_loop {
            quote::quote_spanned! { inner.span() =>
                unsafe {
                    let r = &mut __retained_values.assume_init_mut().#count;
                    Vec::<_>::push(r, #tokens);
                    GPUOwnedResource::__retain(r.last_mut().unwrap())
                }
            }
        } else if self.is_in_divergent_control_flow {
            quote::quote_spanned! { inner.span() =>
                unsafe{
                    let r = &mut __retained_values.assume_init_mut().#count;
                    let old = std::mem::replace(r, Some(#tokens));
                    assert!(old.is_none());
                    GPUOwnedResource::__retain(r.as_mut().unwrap())
                }
            }
        } else {
            quote::quote_spanned! { inner.span() =>
                unsafe{
                    let r = maybe_uninit_new(&mut __retained_values.assume_init_mut().#count);
                    r.write(#tokens);
                    GPUOwnedResource::__retain(r.assume_init_mut())
                }
            }
        }
    }
}

impl VisitMut for GPUFutureTraverser {
    fn visit_stmt_mut(&mut self, i: &mut syn::Stmt) {
        match i {
            syn::Stmt::Macro(inner) if inner.mac.path.is_ident("retain") => {
                *i = syn::Stmt::Expr(
                    syn::Expr::Verbatim(self.transform_retain_macro(&inner.mac)),
                    inner.semi_token,
                );
                return;
            }
            _ => (),
        }

        syn::visit_mut::visit_stmt_mut(self, i);
    }
    fn visit_expr_mut(&mut self, i: &mut syn::Expr) {
        match i {
            syn::Expr::Macro(inner) if inner.mac.path.is_ident("retain") => {
                *i = syn::Expr::Verbatim(self.transform_retain_macro(&inner.mac));
                return;
            }
            syn::Expr::Await(inner) => {
                let count = syn::Index::from(self.retained_value_count);

                if self.is_in_loop {
                    *i = syn::Expr::Verbatim(quote::quote_spanned! { inner.span() =>
                        {
                            let rhyolite::future::GPUFutureBlockReturnValue { output, retained_values } = #inner;
                            Vec::<_>::push(
                                unsafe{&mut (&mut (*__retained_values.as_mut_ptr())).#count},
                                retained_values
                            );
                            output
                        }
                    });
                } else if self.is_in_divergent_control_flow {
                    *i = syn::Expr::Verbatim(quote::quote_spanned! { inner.span() =>
                        {
                            let rhyolite::future::GPUFutureBlockReturnValue { output, retained_values } = #inner;
                            unsafe{(&mut (*__retained_values.as_mut_ptr())).#count = Some(retained_values)};
                            output
                        }
                    });
                } else {
                    *i = syn::Expr::Verbatim(quote::quote_spanned! { inner.span() =>
                        {
                            let rhyolite::future::GPUFutureBlockReturnValue { output, retained_values } = #inner;
                            unsafe{(&mut (*__retained_values.as_mut_ptr())).#count = retained_values};
                            output
                        }
                    });
                }

                self.retained_value_count += 1;
                return;
            }
            _ => (),
        }
        syn::visit_mut::visit_expr_mut(self, i);
    }
    fn visit_expr_await_mut(&mut self, i: &mut syn::ExprAwait) {
        syn::visit_mut::visit_expr_await_mut(self, i);
    }

    fn visit_expr_if_mut(&mut self, i: &mut syn::ExprIf) {
        let b4 = self.is_in_divergent_control_flow;
        self.is_in_divergent_control_flow = true;
        syn::visit_mut::visit_expr_if_mut(self, i);
        self.is_in_divergent_control_flow = b4;
    }
    fn visit_expr_for_loop_mut(&mut self, i: &mut syn::ExprForLoop) {
        let b4 = self.is_in_divergent_control_flow;
        let b5 = self.is_in_loop;
        self.is_in_divergent_control_flow = true;
        self.is_in_loop = true;
        syn::visit_mut::visit_expr_for_loop_mut(self, i);
        self.is_in_divergent_control_flow = b4;
        self.is_in_loop = b5;
    }
    fn visit_expr_while_mut(&mut self, i: &mut syn::ExprWhile) {
        let b4 = self.is_in_divergent_control_flow;
        let b5 = self.is_in_loop;
        self.is_in_divergent_control_flow = true;
        self.is_in_loop = true;
        syn::visit_mut::visit_expr_while_mut(self, i);
        self.is_in_divergent_control_flow = b4;
        self.is_in_loop = b5;
    }
    fn visit_expr_loop_mut(&mut self, i: &mut syn::ExprLoop) {
        let b4 = self.is_in_divergent_control_flow;
        let b5 = self.is_in_loop;
        self.is_in_divergent_control_flow = true;
        self.is_in_loop = true;
        syn::visit_mut::visit_expr_loop_mut(self, i);
        self.is_in_divergent_control_flow = b4;
        self.is_in_loop = b5;
    }
    fn visit_expr_match_mut(&mut self, i: &mut syn::ExprMatch) {
        let b4 = self.is_in_divergent_control_flow;
        self.is_in_divergent_control_flow = true;
        syn::visit_mut::visit_expr_match_mut(self, i);
        self.is_in_divergent_control_flow = b4;
    }
    fn visit_expr_return_mut(&mut self, i: &mut syn::ExprReturn) {
        self.is_in_divergent_control_flow = true;
        syn::visit_mut::visit_expr_return_mut(self, i);
    }
}
